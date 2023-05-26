from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt",
    nargs='+',
    type=str,
    default=None,
    help="the prompt to render"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/txt2img-samples"
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=32,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="logs/f8-kl-clip-encoder-256x256-run1/checkpoints/last.ckpt",
    help="path to checkpoint of model",
)
opt = parser.parse_args()

pipe = StableDiffusionPipeline.from_pretrained(opt.ckpt)

pipe.scheduler.config['prediction_type'] = "v_prediction"
pipe.scheduler.config['num_train_timesteps'] = 1024
pipe.scheduler.config['steps_offset'] = 1024 // opt.ddim_steps - 1
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
print(pipe.scheduler)

disable_safety = True

if disable_safety:
  def null_safety(images, **kwargs):
      return images, False
  pipe.safety_checker = null_safety

prompt = opt.prompt or ['a couple of pokemon standing next to each other', 'a cartoon picture of a bear holding a baseball bat', 'a pink bird sitting on top of a white surface', 'a blue and black object with two eyes']
image = pipe(prompt, num_inference_steps=opt.ddim_steps, guidance_scale=opt.scale,)
os.makedirs(opt.outdir, exist_ok=True)
for i, img in enumerate(image.images):
    img.save(os.path.join(opt.outdir, f'img_{i}.png'))

