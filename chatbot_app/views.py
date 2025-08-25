from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer once
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def chat_page(request):
    return render(request, "chat.html")

@api_view(['POST'])
def chat_api(request):
    user_message = request.data.get('message', '')

    if not user_message:
        return Response({'error': 'No message provided.'}, status=400)

    input_ids = tokenizer.encode(user_message, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return Response({'response': response_text})
