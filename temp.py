import re # regex module

text = """
MedicalAssistant\r\nhowareyoudoing     \n
This is meical advice     \n
ok owkring   no 
"""
text = text.replace(r"\r\n","\n").replace(r"\r", "\n") #replace all the \r\n -> \n then \r->\n
line = [item.rstrip() for item in text.split("\n")] # make a list by spliting with \n
text = "\n".join(line) # join on \n as seperator

print(text)

'''

Why Use It?

• Preserves the Traceback: When the new exception is printed, Python will display both errors, clearly showing that the second error was a direct result of the first one.
• Improves Debugging: It prevents the original root cause from being lost or hidden when you wrap low-level errors into high-level, custom application errors.
• Clean Code: It explicitly states your intent to the interpreter and other developers that the exceptions are linked.

'''
try:
    # Code that might fail
    value = int("not_a_number")
except ValueError as exe:
    # Raise a new error and chain the original 'exe' to it
    raise RuntimeError("Failed to process data") from exe
