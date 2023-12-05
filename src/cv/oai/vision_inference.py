# FARLAB - UrbanECG 
# Developer: @mattwfranchi
# Last Edited: 11/08/2023

# This script houses a class to establish a session with the OpenAI GPT4-Vision API, and run inference on a set of images. 

# Module Imports
import os 
import sys 
import json

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
)

import base64

import aiofiles
import asyncio 
import aiohttp
from aiohttp import ClientTimeout

from traceback import TracebackException 
from types import TracebackType


from user.params.oai_creds import OAI_KEY
from user.params.oai import VISION_URL
from user.params.oai import make_headers, make_payload

from src.utils.logger import setup_logger


class OAI_Session: 
    def __init__(self, api_key=OAI_KEY) -> None: 
        self.log = setup_logger("OpenAI Session")
        self.api_key = api_key
        
        self.log.setLevel("INFO")
        self.headers = make_headers(self.api_key)

        self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=5))
        self.log.success("Initialized session.")
    
    async def __aenter__(self) -> "OAI_Session":
        return self
    
    async def __aexit__(
                        self, 
                        exc_type: Exception, 
                        exc_val: TracebackException, 
                        traceback: TracebackType,
                        ) -> None: 
        await self.close()
    
    async def close(self) -> None: 
        await self._session.close()
    
    async def __post(self, url, headers, payload): 
        try: 
            async with self._session.request("POST", url, headers=headers, json=payload) as response: 
                #print(await response.json())
                response.raise_for_status()
                
                #print(response.json)
                #print(response.status)
                return await response.json()
        except Exception as e: 
            self.log.error(f"Error in POST request: {e}")

        

        

    async def __encode_image(self, image_path): 
        async with aiofiles.open(image_path, mode="rb") as image_file: 
            encoded_image = base64.b64encode(await image_file.read()).decode('utf-8')
        return encoded_image

    def update_headers(self, headers): 
        self.headers = headers 
        self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.log.success("Updated headers.")
    
    async def infer_image(self, img_path, outfile="flooding_gptv.json"): 
        encoded_image = await self.__encode_image(img_path)
        payload = make_payload(encoded_image, img_path)
        
        response =  await self.__post(VISION_URL, self.headers, payload)

        # write response to file async
        async with aiofiles.open(outfile, mode="a") as f: 
            await f.write(json.dumps(response) + "\n")

    
    async def infer_images(self, img_paths, outfile="flooding_gptv.json"):
        tasks = [self.infer_image(img_path, outfile) for img_path in img_paths]
        return await asyncio.gather(*tasks)

        