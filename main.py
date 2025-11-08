import os
import asyncio
import tempfile
import httpx
import yt_dlp
from openai import AsyncOpenAI
from typing import Optional

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, ConfigSchema, ConfigSchemaType
from astrbot.api import logger
from astrbot.api.provider import ProviderManager

# 1. 定义你的插件配置项 (这将自动生成 WebUI)
plugin_config_schema = ConfigSchema(
    version="1.3.0",
    schema={
        "stt_provider_id": ConfigSchemaType.STRING(
            title="STT 提供商 (Whisper)",
            description="选择一个在 AstrBot 全局配置的 STT (语音转文字) 提供商。",
            default="",
            _special="select_provider_stt"  # 关键: 这会生成 STT 下拉菜单
        ),
        "dify_provider_id": ConfigSchemaType.STRING(
            title="LLM 提供商 (Dify)",
            description="选择一个在 AstrBot 全局配置的 Dify (LLM) 提供商。",
            default="",
            _special="select_provider"  # 关键: 这会生成 LLM 下拉菜单
        ),
        "dify_workflow_url": ConfigSchemaType.STRING(
            title="Dify Workflow URL",
            description="Dify 工作流的 API 端点 (必须是 .../v1/workflows/run)",
            default=""
        ),
        "dify_input_variable": ConfigSchemaType.STRING(
            title="Dify 输入变量名",
            description="Dify 工作流 '开始' 节点中接收文本的变量名",
            default="transcript"
        ),
        "dify_answer_key": ConfigSchemaType.STRING(
            title="Dify 回复字段名",
            description="Dify 返回的 JSON 中代表最终答案的字段名",
            default="answer"
        )
    }
)

@register(
    "URLSummarizer", 
    "Xiang Junyi", 
    "通过 yt-dlp, Whisper 和 Dify 从 URL 总结音视频内容", 
    "1.3.0", 
    "YOUR_REPO_URL_HERE",
    config_schema=plugin_config_schema  # 2. 注册你的配置
)
class URLSummarizerPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.config = self.context.get_config("URLSummarizer")
        self.is_configured = False
        self.openai_client = None
        self.dify_api_key = None
        self.dify_workflow_url = self.config.get("dify_workflow_url")
        self.dify_input_variable = self.config.get("dify_input_variable", "transcript")
        self.dify_answer_key = self.config.get("dify_answer_key", "answer")
        self.httpx_client = httpx.AsyncClient(timeout=300.0)

        try:
            # 3. 从配置中读取被选中的 "ID"
            stt_provider_id = self.config.get("stt_provider_id")
            dify_provider_id = self.config.get("dify_provider_id")

            if not all([stt_provider_id, dify_provider_id, self.dify_workflow_url]):
                logger.error("URLSummarizer: 插件配置不完整。请在 WebUI 中选择提供商并填写 URL。")
                return

            provider_manager: ProviderManager = self.context.provider_manager
            
            # 4. 查找并配置 STT (Whisper)
            # _special="select_provider_stt" 返回的是 ID
            stt_provider_inst = provider_manager.get_provider_instance(stt_provider_id)
            if stt_provider_inst and hasattr(stt_provider_inst, "api_key"):
                self.openai_client = AsyncOpenAI(api_key=stt_provider_inst.api_key)
            else:
                logger.error(f"URLSummarizer: 未找到 ID 为 '{stt_provider_id}' 的 STT 提供商, 或其缺少 'api_key'。")
                return

            # 5. 查找并配置 Dify
            # _special="select_provider" 返回的是 ID
            dify_provider_inst = provider_manager.get_provider_instance(dify_provider_id)
            if dify_provider_inst and hasattr(dify_provider_inst, "api_key"):
                self.dify_api_key = dify_provider_inst.api_key
            else:
                logger.error(f"URLSummarizer: 未找到 ID 为 '{dify_provider_id}' 的 Dify 提供商, 或其缺少 'api_key'。")
                return

            self.is_configured = True
            logger.info("URLSummarizer: 插件已成功加载并配置。")

        except Exception as e:
            logger.error(f"URLSummarizer: 初始化失败: {e}")

    @filter.command("sum_url")
    async def summarize_url_handler(self, event: AstrMessageEvent, url: str):
        
        if not self.is_configured:
            yield event.plain_result("URLSummarizer 插件未配置或配置错误。请检查 WebUI 设置和日志。")
            return

        audio_path = None
        try:
            yield event.plain_result(f"收到 URL。正在处理音频...")
            
            audio_path = await self._download_audio(url)
            
            yield event.plain_result(f"音频已下载。正在转录...")
            
            transcript = await self._transcribe_audio(audio_path)
            
            yield event.plain_result(f"文稿已生成。正在总结...")
            
            summary = await self._summarize_text(transcript)
            
            yield event.plain_result(f"总结:\n{summary}")

        except Exception as e:
            logger.error(f"URLSummarizer: 处理 {url} 失败。 错误: {e}")
            yield event.plain_result(f"处理 URL 失败。 错误: {str(e)}")
        
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.error(f"URLSummarizer: 清理临时文件 {audio_path} 失败。 错误: {e}")

    async def _download_audio(self, url: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmpfile:
            file_path = tmpfile.name
        
        ydl_opts = {
            'format': 'm4a-best/bestaudio/best',
            'outtmpl': file_path,
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }],
        }
        
        loop = asyncio.get_running_loop()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            await loop.run_in_executor(None, ydl.download, [url])
        
        return file_path

    async def _transcribe_audio(self, file_path: str) -> str:
        with open(file_path, "rb") as audio_file:
            transcription = await self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcription.text

    async def _summarize_text(self, text: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.dify_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": {
                self.dify_input_variable: text
            },
            "response_mode": "blocking",
            "user": "astrbot-url-summarizer"
        }
        
        response = await self.httpx_client.post(self.dify_workflow_url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        key = self.dify_answer_key
        
        if 'outputs' in data and key in data['outputs']:
            return data['outputs'][key]
        elif key in data:
            return data[key]
        else:
            logger.error(f"Dify API 响应中未找到 '{key}': {data}")
            raise ValueError("Dify API 响应格式不符合预期。")

    async def terminate(self):
        if self.is_configured:
            await self.httpx_client.aclose()