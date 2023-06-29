import json
import torch
import evaluate
from xturing.models import BaseModel

model = BaseModel.create("galactica_lora")
e = torch.load('/home/sslashinin/kovakimyan/xturing/galactica/saved_model/pytorch_model.bin')
m = model.engine.model.load_state_dict(e)

mathqa_data = json.load(open("math_qa.json"))
instructions = []
inputs = []
outputs = []

for data in mathqa_data:
    instructions.append(data["instruction"])
    inputs.append(data["input"])
    outputs.append(data["output"])

data_dict = {
    "test": {"instruction": instructions, "text": inputs, "target": outputs}
}
output_general = []
counter = 0
rouge = evaluate.load("rouge")
for instruction in data_dict['test']["instruction"]:
    
    output = model.generate(texts=[instruction])
    predictions = [output[0]]
    references = [
     [data_dict['test']["target"][counter]]
    ]
    
    results = rouge.compute(predictions=predictions, references=references)

    output = model.generate(texts=[instruction])

    predictions = [output[0]]
    references = [
     [data_dict['test']["target"][counter]]
    ]
    counter += 1

    results = rouge.compute(predictions=predictions, references=references)

    test_dict = {"instruction": instruction, "output" : output, "results" : results}
    json_object = json.dumps(test_dict, indent = 4) 
    output_general.append(json_object)
with open("result.json", "w") as file:
  print("[\n", file=file)
  for i in output_general:
    print(i, file=file)
    print(",", file=file)
  print("]\n", file=file)