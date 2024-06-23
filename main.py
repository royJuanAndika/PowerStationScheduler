import arrr
import random
from pyscript import document


def translate_english(event):
    input_text = document.querySelector("#english")
    english = input_text.value
    output_div = document.querySelector("#output")
    output_div.innerText = arrr.translate(english)


def multiply(event):
    firstNum = int(document.querySelector("#first").value)
    secondNum = int(document.querySelector("#second").value)    
    outputH = document.querySelector("#outputH")
    outputH.innerText = firstNum*secondNum
    
def generateRandomNumber(event):
    outputR = document.querySelector("#outputR")
    outputR.innerText = random.randint(98, 100)
    
    
def greet():
    console.log("woi")
    jancok()