from env import *

fd = open("tr.good","r")
trs = eval(fd.readline())

mapping = dict()
for trace in trs:
  for tr in trace:
    s,a,ss,r = tr
    s_abs = abstract(s)
    if s_abs not in mapping:
      mapping[s_abs] = []
    mapping[s_abs].append(a)

for m_key in mapping:
  print m_key, mapping[m_key]
