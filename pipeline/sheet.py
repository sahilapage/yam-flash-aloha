"""sheet.py DIR OUT N0 N1: gridded frames N0..N1-1 of DIR/frames.json tiled 2 wide at 0.6x, labels in full-res px."""
import sys, json, cv2, numpy as np
d, out, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
recs = json.load(open(f'{d}/frames.json'))[a:b]
tiles = []
for r in recs:
    img = cv2.imread(f"{d}/{r['name']}_raw.png")
    s = 0.6; t = cv2.resize(img, None, fx=s, fy=s)
    for x in range(0, 848, 50):
        cv2.line(t, (int(x * s), 0), (int(x * s), t.shape[0]), (0, 0, 255) if x % 100 == 0 else (0, 200, 255), 1)
        if x % 100 == 0: cv2.putText(t, str(x), (int(x * s) + 2, 10), 0, 0.35, (0, 0, 255), 1)
    for y in range(0, 480, 50):
        cv2.line(t, (0, int(y * s)), (t.shape[1], int(y * s)), (0, 0, 255) if y % 100 == 0 else (0, 200, 255), 1)
        if y % 100 == 0: cv2.putText(t, str(y), (2, int(y * s) - 2), 0, 0.35, (0, 0, 255), 1)
    cv2.putText(t, r['name'], (300, t.shape[0] - 6), 0, 0.5, (255, 255, 0), 2)
    tiles.append(t)
while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
