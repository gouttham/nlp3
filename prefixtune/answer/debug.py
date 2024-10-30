from default import *
import os, sys


basemodel = 'distilgpt2'
table_to_text = TableToText("peft", basemodel=basemodel)
model = AutoModelForCausalLM.from_pretrained(basemodel)
decoder_output = table_to_text.decode(model, '../data/input/small.txt')
print("\n".join(decoder_output))