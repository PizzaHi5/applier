#imports
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph_cua import create_cua
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.types import interrupt
from langgraph.graph.message import add_messages

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

from typing import Annotated, TypedDict
from pydantic import Field, BaseModel

# https://docs.scrapybara.com/api-reference/start
from scrapybara import Scrapybara
from scrapybara.tools import ComputerTool, BashTool, EditTool
from scrapybara.openai import UBUNTU_SYSTEM_PROMPT, OpenAI
# from vllm import LLM, SamplingParams

import os
from dotenv import load_dotenv
load_dotenv()

class State(BaseModel):
    """State class to hold the state of the application."""
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    last_input: str = ''
    # browser: BrowserInstance = Field(default_factory=BrowserInstance)

# model = OpenAI(
#     model="Qwen/Qwen2.5-VL-3B-Instruct", 
#     # base_url="http://localhost:57583/v1",
#     temperature=0.8
#     )

SYS_MSG = """
You're a helpful assistant focused on applying to jobs. The browser you are using is already initialized, and visiting google.com.
You can use the browser to search for jobs and apply to them. Find and apply to jobs based on the applicant's resume.

Notes:
- Fill out as much information as you have available on the resume for the job application. 
- If you get stuck and do not know how to proceed, you can ask the user for help.
- If you need to search for a job, you can use the browser to search for jobs based on the resume.
- You can also use the browser to search for job application forms and fill them out.
- Only use information presented in the resume to fill out the job application. Do not use any other information.
            
The applicant's resume is as follows:
{resume}
"""

# Construct the path to the txt file in the parent folder
file_path = os.path.join(os.path.dirname(__file__), '..', 'resume.txt')
with open(file_path, 'r') as file:
    resume_content = file.read()
SYS_MSG = SYS_MSG.format(resume=resume_content)
print("Current system message:\n", SYS_MSG)

messages = [
    HumanMessage(content="Using the resume information you have available, can you apply to a relevant job for me based on my resume?")
]

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# txt = processor.apply_chat_template()
# Nodes
# def human_feedback(state: State):
#     """Function to get human feedback and pause the instance."""
#     print("Pausing for human input...")
#     # instance.pause()  # Pause the instance
#     value = interrupt(
#         # Any JSON serializable value to surface to the human.
#         # For example, a question or a piece of text or a set of keys in the state
#         {'text_to_revise': state["messages"][-1].content},
#     )
#     return {"messages": value}

# https://github.com/langchain-ai/langgraph-cua-py
# cua_graph = create_cua(prompt=SYS_MSG) # recursion limit = 100 by default, can add auth_state_ids from scrapybara

def main():
    # model_ = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        # "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
        # )
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    # model = AutoModelForCausalLM.from_pretrained(
    #     "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    #     )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    text = processor.apply_chat_template(
        SYS_MSG,tokenzier=False,add_generation_prompt=True
        )
    client = Scrapybara()
    instance = client.start_browser()

    image_input, video_input = process_vision_info(messages)

    inputs = processor(
        text=[text],
        image=image_input,
        video=video_input,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # maybe act_stream() instead of act
    response = client.act(
        model=model,
        tools=[
            ComputerTool(instance=instance),
            BashTool(instance=instance),
            EditTool(instance=instance),
        ],
        system=UBUNTU_SYSTEM_PROMPT + '\n' + SYS_MSG,
        prompt="Using the resume information you have available, can you apply to a relevant job for me based on my resume?",
        on_step=lambda step: print(f"Step:\n{step.text}"),
    )

    messages = response.messages
    steps = response.steps
    text = response.text
    usage = response.usage
    # stream = cua_graph.astream(
    #     input={'messages': messages},
    #     stream_mode="updates"
    # )

    # async for update in stream:
    #     if "create_vm_instance" in update:
    #         print(f"Created VM instance")
    #         stream_url = update.get("create_vm_instance", {}).get("stream_url")
    #         print(f"Stream URL: {stream_url}")
    # print("Done")

if __name__ == "__main__":
    main()

