# Algorithms

## Task
- Detect the shadowed region
- Restore the illumination in that region

## Assumption
- Single primary light source
- Each surface that has shadows has unshadowed part
- The shadow may cover adjacent regions

## Process
1. User indicate the shadow by A mouse click (The only step that need user interaction)
2. Compute the shadowed region
3. Compute 3 masks for:
	1. the lit part(outer part)
	2. the shadowed part(except the lit part within the shadowed part)
	3. the entire shadowed region(includes tha lit part within the shadowed part)
4. Use a Pyramid-based restoration process to produce the shadow-free image
5. Apply image painting along a thin border

## Detection
Outcome: 