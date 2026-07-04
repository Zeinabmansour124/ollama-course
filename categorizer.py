import ollama
import os

model="mistral"

input_file="./data/grocery_list.txt"
output_file="./data/grocery_list_categorized.txt"

#check if the input file exists
if not os.path.exists(input_file):
    print(f"Input file {input_file} does not exist.")
    exit(1)

#read the input file
with open(input_file, 'r') as f:
    grocery_list = f.read().strip()

prompt=f"""
you are an assistnt that categirizes and sorts groceery items 
here is a list of grocery items
{grocery_list}
please : 
1.Categorize the items into appropriate categories such as Produce , Dairy , Meat , Bakery , Beverages , Snacks , Frozen Foods , etc.
2.sort the items within each category in alphabetical order.
3.present the output in a clear and organized format, with each category as a heading followed by the sorted list of items under that category.
"""

try:
    response=ollama.generate(model=model, prompt=prompt)
    generated_text=response.get("response", "")
    print(generated_text)
    #write the output to a file
    with open(output_file, 'w') as f:
        f.write(generated_text.strip())
    print(f"Categorized grocery list has been written to {output_file}")
except Exception as e:
    print(f"An error occurred: {str(e)}")