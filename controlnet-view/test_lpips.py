import lpips

loss_fn = lpips.LPIPS(net='alex', version='0.1')

img0 = lpips.im2tensor(lpips.load_image("image_log/val/gs-00000000/reconstruction_gs-00000000.png"))
img1 = lpips.im2tensor(lpips.load_image("image_log/val/gs-00000000/target_view_gs-00000000.png"))

lpips_dis = loss_fn.forward(img0, img1)

print(float(lpips_dis))
