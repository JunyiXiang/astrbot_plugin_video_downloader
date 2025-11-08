import os
import asyncio
import tempfile
import httpx
import yt_dlp
from openai import AsyncOpenAI

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger

class URLSummarizerPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.dify_api_key = os.environ.get("DIFY_API_KEY")
        self.dify_api_url = os.environ.get("DIFY_API_URL")

        if not all([self.openai_api_key, self.dify_api_key, self.dify_api_url]):
            logger.error("URLSummarizer: Missing one or more API keys or URL.")
            raise ValueError("API keys and URL must be set in environment variables.")

        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        self.httpx_client = httpx.AsyncClient(timeout=300.0)

    @filter.command("sum_url")
    async def summarize_url_handler(self, event: AstrMessageEvent, url: str):
        
        audio_path = None
        try:
            yield event.plain_result(f"Received URL. Processing audio...")
            
            audio_path = await self._download_audio(url)
            
            yield event.plain_result(f"Audio downloaded. Transcribing...")
            
            transcript = await self._transcribe_audio(audio_path)
            
            yield event.plain_result(f"Transcript complete. Summarizing...")
            
            summary = await self._summarize_text(transcript)
            
            yield event.plain_result(f"Summary:\n{summary}")

        except Exception as e:
            logger.error(f"URLSummarizer: Failed to process {url}. Error: {e}")
            yield event.plain_result(f"Failed to process URL. Error: {str(e)}")
        
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.error(f"URLSummarizer: Failed to clean up temp file {audio_path}. Error: {e}")

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
                "transcript": text
            },
            "response_mode": "blocking",
            "user": "astrbot-url-summarizer"
        }
        
        response = await self.httpx_client.post(self.dify_api_url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if 'answer' in data:
            return data['answer']
        elif 'outputs' in data and 'answer' in data['outputs']:
            return data['outputs']['answer']
        else:
            logger.error(f"Dify API response missing 'answer': {data}")
            raise ValueError("Dify API response format is not as expected.")

    async def terminate(self):
        await self.httpx_client.aclose()


@register("URLSummarizer", "Xiang Junyi", "Summarizes audio/video from a URL via yt-dlp, Whisper, and Dify.", "1.0.0", "YOUR_REPO_URL_HERE")
def register_plugin(context: Context):
    return URLSummarizerPlugin(context)