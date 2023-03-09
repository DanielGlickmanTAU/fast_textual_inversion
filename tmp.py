
for id, images in d.items():
    if len(images) <= 2:
     os.popen(f'mkdir {id}')
for id, images in d.items():
    if len(images) <= 2:
        os.popen(f'mv {" ".join(images)} {id}')