import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = "neco-ai-lora-v2"

tokenizer = None
model = None


def load_model():
    global tokenizer, model

    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        model.eval()

    return model, tokenizer


def cevapla(message, history):
    try:
        model, tokenizer = load_model()

        messages = [
            {
                "role": "system",
                "content": (
                    "Sen NECO.AI v2 adlı Türkçe yapay zekâ asistanısın. "
                    "Soruları açık, doğru ve anlaşılır şekilde cevapla. "
                    "Kod istendiğinde eksiksiz kod vermeye çalış. "
                    "Emin olmadığın bilgileri kesinmiş gibi söyleme."
                )
            }
        ]

        for user_message, assistant_message in history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_message})

        messages.append({"role": "user", "content": message})

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        answer_tokens = outputs[0][inputs.shape[-1]:]
        answer = tokenizer.decode(
            answer_tokens,
            skip_special_tokens=True
        ).strip()

        return answer or "Bir cevap oluşturamadım. Tekrar dener misin?"

    except Exception as error:
        return f"Hata oluştu: {error}"


demo = gr.ChatInterface(
    fn=cevapla,
    title="NECO.AI v2",
    description="Türkçe sohbet ve kodlama asistanı",
    examples=[
        "Merhaba, sen kimsin?",
        "Arduino nedir?",
        "Python'da basit hesap makinesi kodu yaz."
    ]
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
