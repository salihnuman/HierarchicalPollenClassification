import os
"""
with open('bingol_metadata.txt', 'r') as f:
    lines = f.read().splitlines()

gene_to_family = {}
for line in lines:
    if line.startswith("-"):
        family_name = line[1:].strip()
        gene_to_family[family_name] = family_name
    else:
        gene_to_family[line.split(" ")[0]] = family_name


# Write the gene_to_family mapping to a file
with open('gene_to_family.txt', 'w') as f:
    for gene, family in gene_to_family.items():
        f.write(f"{gene}\t{family}\n")

for filename in os.listdir("BingolPollen_Species"):
    if filename.split(" ")[0] not in gene_to_family:
        print(f"Warning: {filename} not found in gene_to_family mapping.")

"""
with open("gene_to_family.txt", "a") as f:
    f.write(f"Muscari\tAsparagaceae\n")
