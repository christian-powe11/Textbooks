from turtle import Turtle

def turtle_command(text):
  commandList = text.split(",")
  command = commandList[0]
  print (command)
  if command == "goto":
    x = float(commandList[1])
    y = float(commandList[2])
    width = float(commandList[3])
    color = commandList[4].strip()
    t.width(width)
    t.pencolor(color)
    t.goto(x,y)
  elif command == "circle":
    radius = float(commandList[1])
    width = float(commandList[2])
    color = commandList[3].strip()
    t.width(width)
    t.pencolor(color)
    t.circle(radius)
  elif command == "beginfill":
    color = commandList[1].strip()
    t.fillcolor(color)
    t.begin_fill()
  elif command == "endfill":
    t.end_fill()
  elif command == "penup":
    t.penup()
  elif command == "pendown":
    t.pendown()
  else:
    print("Unknown command found in file:",command)

def main():
  # getting txt file's name    
  filename = input("Please enter filename: ")
  
  global t
  t = Turtle()
  
  screen = t.getscreen()
  
  with open(filename) as f:
  
  # with open("lab01/Triforce.txt") as f:
    for line in f:
      text = line.strip()
      turtle_command(text)
  
  t.ht()
  screen.exitonclick()
  
  print("Program Execution Completed.")
 

if __name__ == "__main__":
  # Drawing happens here.
  main()