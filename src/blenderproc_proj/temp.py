# find missing file and patch them only if scene_gt_info.json does not have the value everything else hases
import json
import cv2
import glob

with open("output/bop/train_pbr/000002/scene_gt_info.json") as f:
    info = json.load(f)


gt = json.load(open("output/bop/train_pbr/000002/scene_gt.json"))



def find_missing_entries():
    missing = []
    missing = sorted(set(gt.keys()) - set(info.keys()), key=int)
    print("Missing:", len(missing))
    print("First 20:", missing[:20])
    print("Last 20:", missing[-20:])
    return missing

missing = find_missing_entries()

for obj in missing:

    # if obj < 10 obj = f"00{obj}"
    entries = []
    # if 10 <= obj < 100: f"0{obj}"
    # else obj = obj 
    obj = int(obj)
    if obj < 10:
        obj = f"00{str(obj)}"
    elif 10 <= obj < 100:
        obj = f"0{str(obj)}"
    else:
        obj = str(obj)
    print(obj)


    for f in sorted(glob.glob(f"output/bop/train_pbr/000002/mask_visib/000{obj}_*.png")):
        m = cv2.imread(f, 0)
        print(f"Looking into file: {f}")
        ys, xs = (m > 0).nonzero()

        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())

        bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

        px = int((m > 0).sum())

        entries.append({
            "bbox_obj": bbox,
            "bbox_visib": bbox,
            "px_count_all": px,
            "px_count_valid": px,
            "px_count_visib": px,
            "visib_fract": 1.0
        })

    info[obj] = entries

    print(info[obj])


# with open("output/bop/train_pbr/000002/scene_gt_info.json", "w") as f:
#     json.dump(info, f)



