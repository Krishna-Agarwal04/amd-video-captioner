import os
import json
import logging
from openai import OpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_client")

class StyledCaptions(BaseModel):
    formal: str = Field(description="Professional, structured, and objective description of the video.")
    sarcastic: str = Field(description="Mocking, dry, sarcastic observation of the video events.")
    humorous_tech: str = Field(description="Humorous explanation using tech, coding, database, or developer references.")
    humorous_non_tech: str = Field(description="Everyday casual humor or relatable non-technical jokes.")

class FireworksLLMClient:
    def __init__(self):
        # Read environment variables injected by hackathon environment
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        self.base_url = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        
        # Available models
        # Track 2 lets us pick any model from Fireworks.
        # Default VLM: Kimi K2.6 (multimodal, active on AMD GPU cluster)
        self.vlm_model = os.getenv("VLM_MODEL", "accounts/fireworks/models/kimi-k2p6")
        # Default Text Model: DeepSeek V4 Pro (active on AMD GPU cluster)
        self.text_model = os.getenv("TEXT_MODEL", "accounts/fireworks/models/deepseek-v4-pro")
        
        if not self.api_key:
            logger.warning("FIREWORKS_API_KEY environment variable is not set. API calls will fail.")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        logger.info(f"Initialized FireworksLLMClient. Base URL: {self.base_url}")
        logger.info(f"Using VLM Model: {self.vlm_model} | Text Model: {self.text_model}")

    def describe_frames(self, frames: list[dict]) -> str:
        """
        Sends the base64 frames to a vision-language model to get a factual description.
        Attempts a single-call multi-image request first. Fallback to frame-by-frame aggregation if it fails.
        """
        if not frames:
            return "No frames available to describe."

        # Attempt multi-image description
        try:
            logger.info("Attempting single-call multi-image visual description...")
            
            content = [
                {
                    "type": "text",
                    "text": (
                        "Below are uniformly sampled chronological frames from a video clip. "
                        "Please provide a dense, detailed, and completely factual description of the actions, "
                        "objects, text overlays, and scene transitions visible across these frames. "
                        "Focus on objective facts. Do not write a creative story."
                    )
                }
            ]
            
            for f in frames:
                content.append({"type": "text", "text": f"\n[Frame at {f['timestamp']}]:"})
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{f['base64_image']}"
                    }
                })
                
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.vlm_model,
                messages=messages,
                max_tokens=800,
                temperature=0.2
            )
            description = response.choices[0].message.content.strip()
            logger.info("Successfully generated factual description via single-call multi-image VLM.")
            return description
            
        except Exception as e:
            logger.error(f"Single-call multi-image VLM call failed: {e}. Falling back to sequential frame-by-frame description...")
            return self._describe_frames_sequentially(frames)

    def _describe_frames_sequentially(self, frames: list[dict]) -> str:
        """
        Fallback method that describes each frame individually and joins them into a timeline description.
        """
        frame_descriptions = []
        for i, f in enumerate(frames):
            logger.info(f"Describing frame {i+1}/{len(frames)} (timestamp: {f['timestamp']})...")
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the main activity, text, or objects visible in this single frame from a video clip. Be concise and factual."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{f['base64_image']}"}
                            }
                        ]
                    }
                ]
                response = self.client.chat.completions.create(
                    model=self.vlm_model,
                    messages=messages,
                    max_tokens=150,
                    temperature=0.1
                )
                desc = response.choices[0].message.content.strip()
                frame_descriptions.append(f"At {f['timestamp']}: {desc}")
            except Exception as fe:
                logger.error(f"Failed to describe frame at timestamp {f['timestamp']}: {fe}")
                frame_descriptions.append(f"At {f['timestamp']}: [Frame description failed]")
                
        # Join into a single narrative
        combined_timeline = "\n".join(frame_descriptions)
        logger.info("Successfully completed sequential frame-by-frame narrative compilation.")
        return f"Chronological Video Timeline:\n{combined_timeline}"

    def generate_styled_captions(self, factual_description: str) -> dict:
        """
        Passes the factual description to a Gemma model to generate 4 distinct styled captions in a single call.
        Enforces structured output schema matching the StyledCaptions Pydantic model.
        """
        logger.info("Generating 4 distinct caption styles using Gemma...")
        
        system_prompt = (
            "You are an expert copywriter, stand-up comedian, and principal software engineer. "
            "Your task is to take a chronological factual description of a video and rewrite it into exactly 4 distinct caption/summary styles:\n\n"
            "1. **formal**: A professional, structured, objective, and clear summary of the video. Write in a formal broadcast tone.\n"
            "2. **sarcastic**: A witty, dry, and highly sarcastic observation of the video events. Mock the actions or point out the obvious with dry irony.\n"
            "3. **humorous_tech**: Tech-nerd humor and inside jokes. Use analogies related to programming (bugs, compiling, git merge conflicts, stack overflow, cloud bills, runtime exceptions, server crashes).\n"
            "4. **humorous_non_tech**: Observational, everyday humor. Relatable comedy about daily life (relationships, procrastination, waking up early, coffee addiction) that anyone can understand.\n\n"
            "Make each caption distinct, punchy, and around 1-3 sentences. Return the response ONLY as a JSON object matching the requested schema."
        )
        
        user_prompt = f"Factual Video Description:\n{factual_description}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # We attempt to use JSON schema mode supported by Fireworks AI
            schema = StyledCaptions.model_json_schema()
            
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                response_format={
                    "type": "json_object",
                    "schema": schema
                }
            )
            
            output_text = response.choices[0].message.content.strip()
            parsed = json.loads(output_text)
            logger.info("Successfully generated and parsed structured styled captions.")
            return parsed
            
        except Exception as e:
            logger.warning(f"Structured schema API call or parsing failed: {e}. Retrying with raw JSON prompt fallback...")
            return self._generate_styled_captions_fallback(messages)

    def _generate_styled_captions_fallback(self, messages: list) -> dict:
        """
        Fallback method that requests JSON in text format and uses standard json.loads to parse it.
        """
        # Reconstruct clean messages to avoid consecutive 'user' roles which violates API schemas
        system_content = messages[0]["content"]
        user_content = messages[1]["content"]
        
        fallback_messages = [
            {
                "role": "system", 
                "content": system_content + "\nEnsure you output ONLY a raw JSON object. Do not include markdown code block backticks (```json ... ```)."
            },
            {
                "role": "user", 
                "content": user_content + "\n\nProvide the output strictly as a JSON object with keys: 'formal', 'sarcastic', 'humorous_tech', 'humorous_non_tech'."
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=fallback_messages,
                temperature=0.7,
                max_tokens=1024
            )
            output_text = response.choices[0].message.content.strip()
            
            # Clean markdown JSON formatting if present
            if output_text.startswith("```"):
                lines = output_text.split("\n")
                if lines[0].startswith("```json") or lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                output_text = "\n".join(lines).strip()
                
            parsed = json.loads(output_text)
            logger.info("Successfully recovered and parsed styled captions via fallback.")
            return parsed
        except Exception as fe:
            logger.error(f"Fallback caption generation failed: {fe}")
            # Return dummy values if all fails to prevent entire pipeline crash
            return {
                "formal": "A video clip showcasing the recorded scene.",
                "sarcastic": "Wow, another fascinating video sequence. Absolutely breathtaking.",
                "humorous_tech": "Class VideoCaptioner failed to compile: NullPointerException at runtime.",
                "humorous_non_tech": "When you try to watch a video but the internet provider plays hide and seek."
            }
