from functools import lru_cache
from pathlib import Path
from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repo_architect.config import get_logger, settings
from repo_architect.rag.retriever import HybridCodeRetriever

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model_and_tokenizer(adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        settings.llm_model_id,
        torch_dtype=settings.torch_dtype,
        device_map="auto" if settings.device == "cuda" else None,
    )
    if settings.device == "cpu":
        model = model.to("cpu")
    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def ask_codebase(question: str, repo_id: str, adapter_path: str | None = None) -> Tuple[str, list[str]]:
    docs = HybridCodeRetriever(repo_id=repo_id, k=5).retrieve(question)
    if not docs:
        return "No relevant code blocks found. Ensure the repository is indexed first.", []
    contexts = [doc.page_content for doc in docs]
    messages = [
        {"role": "system", "content": "You are a Senior Software Architect. Answer strictly using the retrieved code snippets and cite exact file paths and line numbers."},
        {"role": "user", "content": f"Retrieved Code:\n{'\n\n---\n\n'.join(contexts)}\n\nQuestion: {question}"},
    ]
    model, tokenizer = load_model_and_tokenizer(adapter_path)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    return answer, contexts
