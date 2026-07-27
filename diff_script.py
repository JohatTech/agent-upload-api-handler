import sys
import difflib
import os

dir1 = sys.argv[1]
dir2 = sys.argv[2]
out_file = sys.argv[3]

with open(out_file, 'w', encoding='utf-8') as f:
    for filename in os.listdir(dir1):
        if not filename.endswith('.py'): continue
        f1 = os.path.join(dir1, filename)
        f2 = os.path.join(dir2, filename)
        if not os.path.exists(f2):
            continue
            
        with open(f1, 'r', encoding='utf-8', errors='replace') as file1:
            lines1 = file1.readlines()
        with open(f2, 'r', encoding='utf-8', errors='replace') as file2:
            lines2 = file2.readlines()
            
        diff = list(difflib.unified_diff(lines1, lines2, f1, f2))
        if diff:
            f.writelines(diff)
            f.write("\n")
