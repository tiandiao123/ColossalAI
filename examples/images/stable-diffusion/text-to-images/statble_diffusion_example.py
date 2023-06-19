from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "The skyling of bellevue is surrounded by clouds"
image = pipe(prompt).images[0]  
    
image.save("example.png")
