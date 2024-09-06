# import os
from enum import Enum
from pathlib import Path

import datasets
import langcodes
import sacrebleu
import typer
from transformers import AutoTokenizer, MistralForCausalLM, PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

app = typer.Typer()

EOI: str = "[/INST]"


def lang_name(lang_code: str) -> str:
    return langcodes.Language(lang_code).language_name()


def chatbot_translate(messages: list[dict[str, str]], model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=20)
    rv =  tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rv[rv.rfind(EOI) + len(EOI):].lstrip()


class TermType(str, Enum):
    term: str = "term"
    rand: str = "rand"
    
@app.command()
def translate(
    outfile: Path, 
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    term_type: TermType = "term",
    verbose: bool = True
) -> None:
    data = datasets.load_dataset("zouharvi/wmt-terminology-2023")["test"]
    translations = []
    model = MistralForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for entry in tqdm(data):
        src_lang, trg_lang = entry["langs"].split("-")
        src_name, trg_name = lang_name(src_lang), lang_name(trg_lang)
        src = entry["src"]
        message = f"Translate the following {src_name} into {trg_name}: {src}."
        message += "You **MUST NOT** provide any explanation in the output other than the=translation itself. "
        dict_entries_text = ", ".join(
            f"{terms[src_lang]} means {terms[trg_lang]}" 
            for terms in entry["hints"][term_type]
        )
        message += f"In this context, {dict_entries_text}. "
        message += f"The full translation in {trg_lang} is:"
        # response = complete(model=model, messages=[{"role": "user", "content": message}])
        messages = [{"role": "user", "content": message}]
        translation = chatbot_translate(messages, model, tokenizer)
        if verbose:
            if not translation:
                print("Empty translation for:", messages)
            else:
                print(f"Translation for message {messages[0]['content']} was:", translation)
        translations.append(translation)
    if verbose:
        print("Writing", len(translations), f"translations to {outfile}")
    outfile.write_text("\n".join(translations))


@app.command()
def evaluate(hypotheses: Path) -> None:
    data = datasets.load_dataset("zouharvi/wmt-terminology-2023")["test"]
    translations = hypotheses.read_text().splitlines()
    refs = [entry["ref"] for entry in data]
    chrf = sacrebleu.CHRF()
    bleu = sacrebleu.BLEU()
    print(chrf.corpus_score(translations, refs))
    print(bleu.corpus_score(translations, refs))
        

if __name__ == "__main__":
    app()
    