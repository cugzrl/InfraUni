model = dict(
    type='ResolveSparseDrive',
    use_grid_mask=True,
    use_deformable_func=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style='pytorch',
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        pretrained=
        '/home/kemove/devdata1/zrl/InfraUni/third_party/SparseDrive/ckpt/resnet50-0676ba61.pth'
    ),
    img_neck=dict(
        type='FPN',
        num_outs=4,
        start_level=0,
        out_channels=256,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048]),
    depth_branch=None,
    head=dict(
        type='ResolveSparseDriveHead',
        task_config=dict(with_det=True, with_map=False, with_motion_plan=True),
        det_head=dict(
            type='Sparse4DHead',
            cls_threshold_to_reg=0.05,
            decouple_attn=True,
            instance_bank=dict(
                type='InstanceBank',
                num_anchor=900,
                embed_dims=256,
                anchor=
                '/home/kemove/devdata1/zrl/InfraUni/data/resolve_sparsedrive/det_anchors.npy',
                anchor_handler=dict(type='SparseBox3DKeyPointsGenerator'),
                num_temp_instances=600,
                confidence_decay=0.6,
                feat_grad=False,
                default_time_interval=0.1,
                max_time_interval=0.51),
            anchor_encoder=dict(
                type='SparseBox3DEncoder',
                vel_dims=3,
                embed_dims=[128, 32, 32, 64],
                mode='cat',
                output_fc=False,
                in_loops=1,
                out_loops=4),
            num_single_frame_decoder=1,
            operation_order=[
                'deformable', 'ffn', 'norm', 'refine', 'temp_gnn', 'gnn',
                'norm', 'deformable', 'ffn', 'norm', 'refine', 'temp_gnn',
                'gnn', 'norm', 'deformable', 'ffn', 'norm', 'refine',
                'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn', 'norm',
                'refine', 'temp_gnn', 'gnn', 'norm', 'deformable', 'ffn',
                'norm', 'refine', 'temp_gnn', 'gnn', 'norm', 'deformable',
                'ffn', 'norm', 'refine'
            ],
            temp_graph_model=dict(
                type='MultiheadAttention',
                embed_dims=512,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            graph_model=dict(
                type='MultiheadAttention',
                embed_dims=512,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            norm_layer=dict(type='LN', normalized_shape=256),
            ffn=dict(
                type='AsymmetricFFN',
                in_channels=512,
                pre_norm=dict(type='LN'),
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            deformable_model=dict(
                type='ResolveFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=4,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='cat',
                kps_generator=dict(
                    type='ResolveBox3DKeyPointsGenerator',
                    num_learnable_pts=6,
                    fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                               [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45],
                               [0, 0, -0.45]])),
            refine_layer=dict(
                type='SparseBox3DRefinementModule',
                embed_dims=256,
                num_cls=10,
                refine_yaw=True,
                with_quality_estimation=True),
            sampler=dict(
                type='SparseBox3DTarget',
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[
                    2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                ],
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                cls_wise_reg_weights=dict()),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_reg=dict(
                type='SparseBox3DLoss',
                loss_box=dict(type='L1Loss', loss_weight=0.25),
                loss_centerness=dict(
                    type='CrossEntropyLoss', use_sigmoid=True),
                loss_yawness=dict(type='GaussianFocalLoss'),
                cls_allow_reverse=[]),
            decoder=dict(type='SparseBox3DDecoder'),
            reg_weights=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            with_instance_id=True),
        motion_plan_head=dict(
            type='ResolveMotionHead',
            fut_ts=10,
            fut_mode=6,
            ego_fut_ts=6,
            ego_fut_mode=6,
            motion_anchor=
            '/home/kemove/devdata1/zrl/InfraUni/data/resolve_sparsedrive/motion_anchors.npy',
            embed_dims=256,
            decouple_attn=True,
            operation_order=[
                'temp_gnn', 'gnn', 'norm', 'ffn', 'norm', 'temp_gnn', 'gnn',
                'norm', 'ffn', 'norm', 'temp_gnn', 'gnn', 'norm', 'ffn',
                'norm', 'refine'
            ],
            temp_graph_model=dict(
                type='MultiheadAttention',
                embed_dims=512,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            graph_model=dict(
                type='MultiheadAttention',
                embed_dims=512,
                num_heads=8,
                batch_first=True,
                dropout=0.1),
            norm_layer=dict(type='LN', normalized_shape=256),
            ffn=dict(
                type='AsymmetricFFN',
                in_channels=256,
                pre_norm=dict(type='LN'),
                embed_dims=256,
                feedforward_channels=512,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            refine_layer=dict(
                type='MotionPlanningRefinementModule',
                embed_dims=256,
                fut_ts=10,
                fut_mode=6,
                ego_fut_ts=6,
                ego_fut_mode=6),
            motion_sampler=dict(type='MotionTarget'),
            motion_loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.2),
            motion_loss_reg=dict(type='L1Loss', loss_weight=0.2),
            motion_decoder=dict(type='SparseBox3DMotionDecoder'),
            num_det=50,
            queue_length=4)))
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_size = (704, 384)
batch_size = 2
accumulation_steps = 4
amp = False
stage1_epochs = 20
stage2_epochs = 10
learning_rate = 0.0001
weight_decay = 0.001
grad_clip = 25.0
warmup_steps = 500
seed = 20261028
camera_order = ['axis1', 'axis2', 'axis3', 'axis4']
protocol = 'resolve_official_curves_paper_nds9_v1'
forecast_steps = 10
forecast_hz = 2
forecast_modes = 6
split_file = '/home/kemove/devdata1/zrl/InfraUni/third_party/MC-DeTra/splits/resolve_forecast_split.json'
