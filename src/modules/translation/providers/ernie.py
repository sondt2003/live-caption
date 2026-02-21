import os
import json
import requests
from ..base import BaseTranslator
from loguru import logger

class ErnieTranslator(BaseTranslator):
    def __init__(self):
        self.api_key = os.getenv('BAIDU_API_KEY')
        self.secret_key = os.getenv('BAIDU_SECRET_KEY')
        self.access_token = None

    def _get_access_token(self):
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.api_key}&client_secret={self.secret_key}"
        response = requests.post(url, headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            raise Exception("Failed to get Baidu access token")

    def translate(self, messages: list, json_mode: bool = True) -> str:
        if self.access_token is None:
            self.access_token = self._get_access_token()
        
        # In Ernie, system message is separate from chat messages
        system_msg = ""
        user_msgs = []
        for msg in messages:
            if msg['role'] == 'system':
                system_msg = msg['content']
            else:
                user_msgs.append(msg)

        model_name = 'ernie-speed-128k'
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model_name}?access_token=" + self.access_token
        payload = json.dumps({
            "messages": user_msgs,
            "system": system_msg
        })
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(url, headers=headers, data=payload)
            if response.status_code == 200:
                response_json = response.json()
                return response_json.get('result')
            else:
                raise Exception(f"Baidu API Error: {response.status_code}")
        except Exception as e:
            logger.error(f"Ernie Translation Error: {e}")
            raise e
