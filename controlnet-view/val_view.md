# val_view.py 的使用

val_view.py 在v100的服务器对应的文件路径

    /DATA/disk1/cihai/lrz/3d-object-reconstruction/controlnet-view

使用的时候激活虚拟环境

    liu-control-1

使用的时候，最好指定（1块）可用的显卡（偷偷用别的，也行吧...）

    # 这就是在用别人的...
    CUDA_VISIBLE_DEVICES=0 python val_view.py

val_view.py 中可以调节的位置

    with torch.no_grad():
        images = model.log_images(batch, ddim_steps=50, unconditional_guidance_scale=19.0)

可以设置ddim_steps，即ddim采样的步数和unconditional_guidance_scale，即cfg的控制值
修改的化可以直接用vim

生成的结果保存在路径

    /DATA/disk1/cihai/lrz/3d-object-reconstruction/controlnet-view/image_log/val

保存的结果中用于对比的内容如下

    control_gs-{batch编号}.png  # 输入的已知视角的图片
    samples_cfg_scale_{cfg的控制值}_gs-{batch编号}  # 重建的图片
    target_view_gs-{batch编号}  # ground truth
