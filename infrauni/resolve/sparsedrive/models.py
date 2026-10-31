"""Retain SparseDrive detection/agent-motion; omit map and ego planning tasks."""
import numpy as np
import torch
from torch import nn
from mmcv.runner import BaseModule
from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear
from mmcv.cnn.bricks.registry import ATTENTION,PLUGIN_LAYERS,FEEDFORWARD_NETWORK,NORM_LAYERS
from mmdet.models import DETECTORS,HEADS,build_loss
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from projects.mmdet3d_plugin.models.sparsedrive import SparseDrive
from projects.mmdet3d_plugin.models.sparsedrive_head import SparseDriveHead
from projects.mmdet3d_plugin.models.motion.motion_planning_head import MotionPlanningHead
from projects.mmdet3d_plugin.models.motion.instance_queue import InstanceQueue
from projects.mmdet3d_plugin.models.attention import gen_sineembed_for_position
from projects.mmdet3d_plugin.models.blocks import linear_relu_ln
from projects.mmdet3d_plugin.models.instance_bank import topk
from projects.mmdet3d_plugin.models.detection3d.decoder import decode_box
from projects.mmdet3d_plugin.models.detection3d.detection3d_blocks import SparseBox3DKeyPointsGenerator
from projects.mmdet3d_plugin.models.blocks import DeformableFeatureAggregation


@PLUGIN_LAYERS.register_module()
class ResolveBox3DKeyPointsGenerator(SparseBox3DKeyPointsGenerator):
    def forward(self,anchor,*args,**kwargs):
        # Limit only the geometry used to sample image features. The raw
        # regression outputs and supervision retain their original values.
        # +/-20 covers 2e-9 to 4.9e8 metres, far beyond any valid object.
        safe_anchor=torch.cat([anchor[...,:3],anchor[...,3:6].clamp(-20.,20.),anchor[...,6:]],dim=-1)
        return super().forward(safe_anchor,*args,**kwargs)


@ATTENTION.register_module()
class ResolveFeatureAggregation(DeformableFeatureAggregation):
    @staticmethod
    def project_points(key_points,projection_mat,image_wh=None):
        extended=torch.cat([key_points,torch.ones_like(key_points[...,:1])],dim=-1)
        projected=(projection_mat[:,:,None,None]@extended[:,None,...,None]).squeeze(-1)
        xy,depth=projected[...,:2],projected[...,2:3]
        visible=(depth>1e-5)&torch.isfinite(projected).all(dim=-1,keepdim=True)
        if image_wh is not None:
            wh=image_wh[:,:,None,None]
            visible=visible&(xy>0).all(dim=-1,keepdim=True)&(xy<wh*depth).all(dim=-1,keepdim=True)
        # Mask before division: an out-of-image point contributes zero in the
        # CUDA operator, but an overflowing unused division can still create
        # NaN derivatives. Valid visible coordinates keep the original math.
        safe_xy=torch.where(visible,xy,torch.zeros_like(xy))
        safe_depth=torch.where(visible,depth,torch.ones_like(depth))
        points=safe_xy/safe_depth
        if image_wh is not None:points=points/image_wh[:,:,None,None]
        return torch.where(visible,points,torch.full_like(points,-1.))

class AgentQueue(InstanceQueue):
    def __init__(self,embed_dims=256,queue_length=4,tracking_threshold=.2):
        # Agent-only subset of the official InstanceQueue. Static roadside sensor.
        nn.Module.__init__(self);self.embed_dims=embed_dims;self.queue_length=queue_length;self.tracking_threshold=tracking_threshold;self.reset()
    def get_agents(self,det_output,metas,mask):
        if self.period is not None and self.period.shape[0]!=det_output['instance_feature'].shape[0]:self.reset()
        self.prepare_motion(det_output,mask)
        features=torch.stack(self.instance_feature_queue,dim=2);anchors=torch.stack(self.anchor_queue,dim=2)
        history=torch.arange(len(self.anchor_queue),0,-1,device=anchors.device)[None,None]
        return features,anchors,history>self.period[...,None]

@HEADS.register_module()
class ResolveMotionHead(MotionPlanningHead):
    def __init__(self,motion_anchor,operation_order,temp_graph_model,graph_model,norm_layer,ffn,refine_layer,motion_sampler,motion_loss_cls,motion_loss_reg,embed_dims=256,fut_ts=10,fut_mode=6,num_det=50,queue_length=4,decouple_attn=True,**unused):
        BaseModule.__init__(self)
        self.embed_dims=embed_dims;self.fut_ts=fut_ts;self.fut_mode=fut_mode;self.num_det=num_det;self.decouple_attn=decouple_attn;self.operation_order=operation_order
        self.instance_queue=AgentQueue(embed_dims,queue_length,.2)
        self.motion_sampler=build_from_cfg(motion_sampler,BBOX_SAMPLERS)
        self.motion_loss_cls=build_loss(motion_loss_cls);self.motion_loss_reg=build_loss(motion_loss_reg)
        configs={'temp_gnn':(temp_graph_model,ATTENTION),'gnn':(graph_model,ATTENTION),'norm':(norm_layer,NORM_LAYERS),'ffn':(ffn,FEEDFORWARD_NETWORK),'refine':(refine_layer,PLUGIN_LAYERS)}
        self.layers=nn.ModuleList([build_from_cfg(*configs[op]) for op in operation_order])
        for op,layer in zip(operation_order,self.layers):
            if op=='refine':
                for name in ['plan_cls_branch','plan_reg_branch','plan_status_branch']:delattr(layer,name)
        self.fc_before=nn.Linear(embed_dims,embed_dims*2,bias=False) if decouple_attn else nn.Identity()
        self.fc_after=nn.Linear(embed_dims*2,embed_dims,bias=False) if decouple_attn else nn.Identity()
        self.motion_anchor=nn.Parameter(torch.tensor(np.load(motion_anchor),dtype=torch.float32),requires_grad=False)
        assert self.motion_anchor.shape==(10,6,10,2)
        self.motion_anchor_encoder=nn.Sequential(*linear_relu_ln(embed_dims,1,1),Linear(embed_dims,embed_dims))

    def init_weights(self):
        for op,layer in zip(self.operation_order,self.layers):
            if op=='refine':nn.init.constant_(layer.motion_cls_branch[-1].bias,-4.59511985)
            else:
                for p in layer.parameters():
                    if p.dim()>1:nn.init.xavier_uniform_(p)

    def forward(self,det_output,map_output,feature_maps,metas,anchor_encoder,mask,anchor_handler):
        feature=det_output['instance_feature'];pos=det_output['anchor_embed'];bs,n,dim=feature.shape
        confidence=det_output['classification'][-1].sigmoid().max(-1).values
        _,(selected,selected_pos)=topk(confidence,min(self.num_det,n),feature,pos)
        hist,hist_anchor,hist_mask=self.instance_queue.get_agents(det_output,metas,mask)
        hist_pos=anchor_encoder(hist_anchor).flatten(0,1);hist=hist.flatten(0,1);hist_mask=hist_mask.flatten(0,1)
        anchors=self.get_motion_anchor(det_output['classification'][-1].sigmoid(),det_output['prediction'][-1])
        mode_query=self.motion_anchor_encoder(gen_sineembed_for_position(anchors[...,-1,:]))
        classifications=[];predictions=[]
        for i,op in enumerate(self.operation_order):
            if op=='temp_gnn':
                feature=self.graph_model(i,feature.flatten(0,1).unsqueeze(1),hist,hist,query_pos=pos.flatten(0,1).unsqueeze(1),key_pos=hist_pos,key_padding_mask=hist_mask).reshape(bs,n,dim)
            elif op=='gnn':feature=self.graph_model(i,feature,selected,selected,query_pos=pos,key_pos=selected_pos)
            elif op in ('norm','ffn'):feature=self.layers[i](feature)
            elif op=='refine':
                q=mode_query+(feature+pos).unsqueeze(2);layer=self.layers[i]
                classifications.append(layer.motion_cls_branch(q).squeeze(-1));predictions.append(layer.motion_reg_branch(q).reshape(bs,n,self.fut_mode,self.fut_ts,2))
        self.instance_queue.cache_motion(feature,det_output,metas)
        return dict(classification=classifications,prediction=predictions),None

    def loss(self,motion_out,planning_out,data,motion_cache):
        # Official motion supervision and mode matching, no fake planning labels.
        return self.loss_motion(motion_out,data,motion_cache)

@HEADS.register_module()
class ResolveSparseDriveHead(SparseDriveHead):
    def loss(self,model_outs,data):
        # Match the official FP16 runner's force_fp32 loss path when using the
        # native PyTorch AMP training loop. Keep focal/quality math in FP32.
        def fp32(value):
            if torch.is_tensor(value):return value.float() if value.is_floating_point() else value
            if isinstance(value,dict):return {k:fp32(v) for k,v in value.items()}
            if isinstance(value,(tuple,list)):return type(value)(fp32(v) for v in value)
            return value
        with torch.cuda.amp.autocast(enabled=False):
            outputs=fp32(model_outs)
            losses=super().loss(outputs,data)
            if getattr(self,'collect_training_diagnostics',False):
                self.last_training_diagnostics=self.training_diagnostics(outputs,data,losses)
            return losses

    @torch.no_grad()
    def training_diagnostics(self,model_outs,data,losses):
        """Report matched objects separately from unconstrained background queries.

        Read the assignment already made by the real loss. Do not run another
        assignment, alter targets, consume random numbers or add loss terms.
        """
        det=model_outs[0];pred=det['prediction'][-1].detach()
        score=det['classification'][-1].detach().sigmoid().max(-1).values
        encoded=self.det_head.sampler.encode_reg_target(data['gt_bboxes_3d'],pred.device)
        matched_pred=[];matched_target=[];matched_score=[]
        for b,(pi,ti) in enumerate(self.det_head.sampler.indices):
            if pi is None or not len(pi):continue
            matched_pred.append(pred[b,pi,:10]);matched_target.append(encoded[b][ti,:10]);matched_score.append(score[b,pi])
        result=dict(queries=int(score.numel()),matched_gt=sum(len(x) for x in matched_pred),class_score_ge_01=int((score>=.1).sum()),class_score_ge_01_extreme_size=int(((score>=.1)&(pred[...,3:6].abs()>10).any(-1)).sum()))
        if not matched_pred:return result
        p=torch.cat(matched_pred);target=torch.cat(matched_target);s=torch.cat(matched_score)
        diff=p-target;regressed=s>self.det_head.cls_threshold_to_reg
        result.update(matched_center_xy_error_m=float(diff[:,:2].norm(dim=-1).mean()),matched_velocity_xy_l1_mps=float(diff[:,8:10].abs().mean()),matched_log_size_min=float(p[:,3:6].min()),matched_log_size_max=float(p[:,3:6].max()),regressed_gt=int(regressed.sum()))
        soft=torch.exp(-diff[regressed,:3].norm(dim=-1)).clamp(1e-7,1-1e-7)
        floor=(-(soft*soft.log()+(1-soft)*(1-soft).log())).sum()/len(p)
        suffix=str(len(det['prediction'])-1)
        result.update(cns_last_decoder_loss=float(losses['det_loss_cns_'+suffix].detach()),cns_last_decoder_entropy_floor=float(floor))
        return result

    def post_process(self,model_outs,data):
        det,_,motion,_=model_outs
        prob=det['classification'][-1].sigmoid();score,label=prob.max(-1)
        quality=det.get('quality')
        out=[]
        for b in range(len(score)):
            # Preserve official SparseBox3DDecoder: top-k by classification,
            # then quality re-score and sort, using one shared query index.
            values,index=torch.topk(score[b],min(300,score.shape[1]),sorted=True)
            if quality is not None and quality[-1] is not None:
                values=values*quality[-1][b,index,0].sigmoid()
                order=values.argsort(descending=True);values=values[order];index=index[order]
            boxes=decode_box(det['prediction'][-1][b,index])
            result=dict(boxes_3d=boxes.detach().cpu(),scores_3d=values.detach().cpu(),labels_3d=label[b,index].detach().cpu())
            if motion is not None:
                future=motion['prediction'][-1][b,index].cumsum(-2)+boxes[:,None,None,:2]
                mode=motion['classification'][-1][b,index].sigmoid();mode=mode/mode.sum(-1,keepdim=True).clamp_min(1e-8)
                result.update(trajs_3d=future.detach().cpu(),trajs_score=mode.detach().cpu())
            out.append(result)
        return out

@DETECTORS.register_module()
class ResolveSparseDrive(SparseDrive):
    def reset_memory(self):
        self.head.det_head.instance_bank.reset()
        if hasattr(self.head.det_head.sampler,'dn_metas'):self.head.det_head.sampler.dn_metas=None
        if hasattr(self.head,'motion_plan_head'):self.head.motion_plan_head.instance_queue.reset()
        self._resolve_scenes=None;self._resolve_times=None
    def forward(self,img,**data):
        scenes=tuple(d['scene_token'] for d in data['img_metas'])
        times=data['timestamp'].detach().cpu().numpy()
        if getattr(self,'_resolve_scenes',None)!=scenes or (getattr(self,'_resolve_times',None) is not None and np.any(times<=self._resolve_times)):
            self.reset_memory()
        self._resolve_scenes=scenes;self._resolve_times=times
        bank=self.head.det_head.instance_bank
        if bank.cached_anchor is not None:
            cached=bank.cached_anchor
            invalid=(~torch.isfinite(cached).all(-1))|(cached[...,3:6].abs()>20.).any(-1)
            if invalid.any():
                # Reinitialize numerically unusable temporal proposals. No
                # GT or class-specific range is used by this safeguard.
                count=cached.shape[1]
                bank.cached_anchor=torch.where(invalid[...,None],bank.anchor.detach()[None,:count].expand_as(cached),cached)
                bank.cached_feature=torch.where(invalid[...,None],torch.zeros_like(bank.cached_feature),bank.cached_feature)
                if bank.confidence is not None:bank.confidence=bank.confidence.masked_fill(invalid,0.)
                if bank.instance_id is not None:
                    bank.instance_id=torch.cat([bank.instance_id[:,:count].masked_fill(invalid,-1),bank.instance_id[:,count:]],dim=1)
                if hasattr(self,'numeric_watch'):self.numeric_watch['invalid_cached_proposals_reset']=int(invalid.sum())
        return super().forward(img,**data)
