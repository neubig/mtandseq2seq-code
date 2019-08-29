import sys
import numpy as np

print('<html><head><title>Probability Visualization</title></head><body>')

def num_to_cell(prob):
  num = max(-20, np.log(prob) if prob else -1e10) / -20.0
  bgcolor = '{:02x}{:02x}{:02x}'.format(int(255-num*255), int(255-num*255), int(255-num*126))
  fcolor = 'FFFFFF' if num > 0.5 else '000000'
  return f'<td bgcolor="#{bgcolor}"><font color="#{fcolor}">{prob:.2e}</font></td>'


curr_lines = []
for line in sys.stdin:
  line = line.strip()
  if line:
    curr_lines.append(line.split(' '))
  else:
    assert(len(curr_lines) > 0)
    depth = len(curr_lines[0])
    print('<table><tr><td>'+'</td><td>'.join([x[0] for x in curr_lines])+'</td></tr>')
    for i in range(1, depth):
      print('<tr>' + ''.join([num_to_cell(float(x[i])) for x in curr_lines]) + '</tr>')
    print('</table><br/><br/>')
    curr_lines = []

print('</html>')
