def split_ribbon(ribbon_path):
    import cv2
    from pathlib import Path

    NUM_SLOTS = 5
    SLOTS_DIR = Path("data/processed/slots")
    SLOTS_DIR.mkdir(parents=True, exist_ok=True)

    ribbon = cv2.imread(str(ribbon_path))
    h, w, _ = ribbon.shape
    slot_width = w // NUM_SLOTS

    print(f"[INFO] Ribbon size: {w}x{h}")

    slot_paths = []
    for i in range(NUM_SLOTS):
        x1 = i * slot_width
        x2 = (i + 1) * slot_width
        slot = ribbon[:, x1:x2]

        slot_path = SLOTS_DIR / f"{ribbon_path.stem}_slot{i}.png"
        cv2.imwrite(str(slot_path), slot)
        slot_paths.append(str(slot_path))

        print(f"[OK] Saved slot {i} → {slot_path}")

    return slot_paths
