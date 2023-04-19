import sys
sys.path.append(r'/DATA/disk1/cihai/lrz/3d-object-reconstruction/controlnet-view')
from ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder

def print_nice(tensor):
    for i in range(77):
        print(tensor[0, i])

if __name__ == '__main__':
    print('test FrozenOpenCLIPEmbedder')
    test_clip_text = FrozenOpenCLIPEmbedder(device="cpu", freeze=True, layer="penultimate")

    prompt0 = test_clip_text.encode("")
    prompt1 = test_clip_text.encode("Professional high-quality wide-angle digital art of a")
    prompt2 = test_clip_text.encode(
        "photorealistic, extremely high detail, cinematic lighting, trending on artstation, cgsociety,"
        "realistic rendering of Unreal Engine 5, 8k, 4k, HQ"
    )

    print_nice(prompt0)
    print(prompt0.shape)
    print_nice(prompt1)
    print(prompt1.shape)
    print_nice(prompt2)
    print(prompt2.shape)
