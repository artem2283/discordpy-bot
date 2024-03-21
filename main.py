import discord
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import numpy as np
import io
import asyncio
import nest_asyncio
import cv2

nest_asyncio.apply()

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

class MyBot(discord.Client):
    intents = discord.Intents.default()
    intents.message_content = True

    async def on_ready(self):
        print(f'We have logged in as {self.user}')

    async def on_message(self, message):
        if message.author == self.user:
            return

        if message.content.startswith('!generate'):
            prompt = message.content.split('!generate ',1)[1]
            image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
            # Преобразуем изображение в формат PIL и сохраняем его в байтовом потоке
            image_np = np.array(image)
            image_np = cv2.bitwise_not(image_np)  # Используйте image_np здесь
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            byte_arr = io.BytesIO()
            image_pil.save(byte_arr, format='PNG')
            byte_arr.seek(0)
            # Отправляем изображение обратно в чат Discord
            await message.channel.send(file=discord.File(byte_arr, 'generated_image.png'))

intents = discord.Intents.default()
intents.message_content = True
bot = MyBot(intents=intents)

bot.run("DISCORD_TOKEN")
