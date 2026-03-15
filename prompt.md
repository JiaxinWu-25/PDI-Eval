为了最大化发挥 **PDI-Eval** 的审计能力，这些 Prompt 经过专门设计，旨在诱导 AI 模型在处理**深度变化、运动解耦、刚性维持**等物理规律时产生“潜在冲突”。

以下是为您设计的 6 个场景，每个场景 10 个 Prompt：

---

### 场景 1：经典纵向收敛 (Classic Longitudinal Convergence)
*   **审计重点**：$h \cdot Z$ 守恒律与 $1/Z^2$ 缩放节奏。
*   **Prompt 核心**：强调直线路径和远方的消失点。

1. A high-speed train traveling away from the camera on a perfectly straight railway track toward the horizon, vanishing point focus.
2. A red vintage car driving away on a long, straight desert highway, cinematic wide shot.
3. A basketball rolling directly away from the camera down a very long, narrow subway corridor.
4. A large cargo ship moving slowly away from the harbor into the open sea, seen from a fixed pier.
5. An airplane taxiing away on a vast, straight runway at sunset, heat haze effect.
6. A cyclist riding away on a straight forest path, trees creating a natural perspective tunnel.
7. A toy car being pushed away on a long wooden dining table, macro perspective.
8. A professional runner sprinting away on a straight 100m track in a stadium.
9. A heavy truck driving away on a straight road during a light snowfall, clear tail lights.
10. A futuristic drone flying straight away through a long, illuminated futuristic tunnel.

---

### 场景 2：动态背景下的跟随 (Dynamic Tracking Shot)
*   **审计重点**：相机位移产生的视差与物体位移的解耦（检测“贴图滑步”）。
*   **Prompt 核心**：手持感、侧向平移、背景流动。

1. A handheld tracking shot following a woman walking down a busy city sidewalk, camera stays at a constant distance.
2. A side-view tracking shot from a moving car, passing a cyclist riding at the same speed.
3. A low-angle following shot of a dog running through a park, grass blurred in the background.
4. A smooth gimbal shot following a skateboarder performing on a long, straight concrete path.
5. A tracking shot from a train window, keeping pace with a car driving on a parallel road.
6. A handheld camera following a man running through a crowded market, noticeable camera shake.
7. A drone following a speedboat from a low altitude, water splashes creating complex foreground.
8. A side-tracking shot of a horse galloping across a flat prairie, distant mountains in the background.
9. A following shot behind a person pushing a shopping cart through a large, empty parking lot.
10. A cinematic tracking shot following a robot walking through a high-tech factory hallway.

---

### 场景 3：非刚性生物运动 (Non-rigid Biological Motion)
*   **审计重点**：$\sigma(R_{integrity})$ 结构协同性（检测肢体拉面化/溶解）。
*   **Prompt 核心**：大幅度肢体动作、生物体、远距离位移。

1. A tall person performing an expressive contemporary dance while moving away from the camera in an open field.
2. A golden retriever jumping and playing while running away on a sandy beach.
3. A person doing a series of cartwheels across a long, empty gymnasium floor.
4. A large elephant walking away slowly in the African savanna, ears flapping.
5. A breakdancer performing a power move and then crawling away from the lens.
6. A group of children playing tag and running in different directions in a park.
7. A ballet dancer doing pirouettes while moving diagonally toward the vanishing point.
8. A monkey swinging between ropes and moving away from the camera in a jungle.
9. A person in a bulky spacesuit walking away on the dusty lunar surface.
10. A giant spider robot with many legs walking away on a rocky terrain.

---

### 场景 4：旋转与复杂姿态 (Rotation & Complex Pose)
*   **审计重点**：自适应修正逻辑（检测旋转中的 2D 投影伪影）。
*   **Prompt 核心**：多轴旋转、转弯、非对称形状。

1. A wooden crate floating in mid-air and slowly rotating on all axes, studio lighting.
2. A car performing a sharp U-turn at a crossroad while moving away from the camera.
3. A drone performing a slow 360-degree spin while flying away into the clouds.
4. A person carrying a large, irregular-shaped mirror while walking and turning around.
5. A futuristic spacecraft tumbling slowly while drifting away into deep space.
6. A large soccer ball rolling and spinning away on a green field.
7. A gymnast performing a flip in mid-air while the camera moves backward.
8. A rotating glass pyramid on a pedestal, reflecting light while moving away on a conveyor belt.
9. A forklift turning and driving away in a narrow warehouse aisle.
10. A bird spiraling upward and away from the camera in the sky.

---

### 场景 5：遮挡物 (Occlusion)
*   **审计重点**：物体常驻性与尺度跳变（检测“出洞瞬间”的初始化错误）。
*   **Prompt 核心**：进入/离开掩体、完全遮挡再现。

1. A car driving behind a series of large concrete pillars and emerging on the other side.
2. A person walking behind a thick brick wall and reappearing after 3 seconds.
3. A train entering a short tunnel and emerging, maintaining a constant speed.
4. A cat walking behind a row of parked cars and jumping out at the end.
5. A cyclist passing behind a large billboard on a city street.
6. A child running behind a thick oak tree in a garden and coming out the other side.
7. A bus obscured by a passing truck and then becoming visible again.
8. A person walking through a dense patch of fog and reappearing further away.
9. A boat sailing behind a small rocky island and emerging on the far side.
10. A remote-controlled car driving under a low sofa and coming out from the back.

---

### 场景 6：希区柯克变焦 (Dolly Zoom)
*   **审计重点**：内参 ($f$) 与外参 ($Z$) 的极端耦合（测试世界模型的深度理解）。
*   **Prompt 核心**：Dolly Zoom 术语、背景扭曲、主体恒定。

1. A classic dolly zoom shot of a man standing still in the center of a narrow, dark alleyway.
2. A dramatic Hitchcock zoom on a lighthouse by the sea, the background stretching while the tower stays the same size.
3. A dolly zoom effect on a woman sitting at a long dinner table, the room elongating behind her.
4. A cinematic dolly zoom on a lone tree in a vast desert, the horizon rushing toward the viewer.
5. A vertigo-effect shot of a person looking down a steep spiral staircase.
6. A dolly zoom on a medieval castle, the mountains behind it growing larger and closer.
7. A fast dolly zoom on a statue in a museum, emphasizing the surrounding architecture distortion.
8. A slow, haunting dolly zoom on an old telephone booth in the middle of a forest.
9. A professional dolly zoom shot on a goalkeeper during a high-stakes penalty kick.
10. A creative dolly zoom on a cup of coffee on a cafe table, the street outside warping.

---

### 🚀 实验建议流程：
1.  **分批生成**：先从每个类别中挑选 2 个最核心的 Prompt（共 12 个），在 Sora/Kling/Luma 上运行。
2.  **建立 ID 索引**：给生成的视频命名为 `S1_P1_Kling.mp4`（场景1_Prompt1_模型名）。
3.  **运行 PDI 审计**：记录下每类场景的平均 PDI 分数。
4.  **寻找“物理冠军”**：对比哪个模型在“场景 6 (Dolly Zoom)”这种最难的几何题上得分最高。

这些 Prompt 的描述非常具体，能有效防止 AI 用简单的背景模糊来逃避几何计算。祝你的实验进展顺利！


---
# 以上是不含要求的，下面是含要求的：

根据您的最新要求（相机位姿变动、特定对象限制、环境复杂度及遮挡时间控制），我为您重新设计了这 6 个场景的各 10 个 Prompt。

这些 Prompt 专门设计为 **PDI-Eval** 的压力测试用例，旨在诱导 AI 模型在处理**动态相机视角下的几何连续性**时产生幻觉。

---

### 场景 1：刚性物体纵向收敛 (Axial Motion, Rigid Only)
*   **要求**：相机手持或移动，无动物，仅限刚体（车辆、机械等）。
*   **审计重点**：$h \cdot Z$ 守恒律与 $1/Z^2$ 缩放节奏。

1. A handheld camera follows a red vintage car driving away on a straight desert highway, subtle camera shake.
2. A high-speed train moving toward the camera on a straight track, seen from a low-angle handheld perspective.
3. A large industrial drone flying away at a steady speed down a long, narrow factory corridor with moving camera.
4. A yellow school bus driving away on a straight suburban street, camera tracking from a low position behind.
5. A silver metallic sphere rolling away on a long, reflective marble floor, camera following closely.
6. An airplane taxiing away on a vast runway, seen from a vibrating camera mounted on a ground vehicle.
7. A heavy cargo truck moving away on a straight bridge at night, camera swaying slightly.
8. A futuristic robot rover moving away across a flat lunar landscape, handheld camera tracking its path.
9. A motorcycle speeding away on a long, straight coastal road, cinematic camera following.
10. A large shipping container being pushed away on a straight industrial dock, moving camera perspective.

---

### 场景 2：非人生物动态跟随 (Tracking Shot, Non-human Bio)
*   **要求**：相机侧向平移或跟随，审计对象为非人生物。
*   **审计重点**：视差解耦与“贴图滑步”检测。

1. A side-view tracking shot from a car window, keeping pace with a horse galloping across a flat prairie.
2. A low-altitude drone following a cheetah sprinting through the tall grass of the African savanna.
3. A handheld camera following a golden retriever running through a park, blurred trees in the background.
4. A smooth tracking shot following a large eagle flying parallel to a cliffside at high speed.
5. A side-tracking shot of a deer jumping across a forest clearing, camera moving through the trees.
6. A close-up tracking shot of a tiger walking through a jungle, camera moving at the same pace.
7. A tracking shot from a boat, following a dolphin swimming and jumping in the parallel waves.
8. A low-angle camera following a cat running along a straight brick wall in an alleyway.
9. A side-view tracking shot of a squirrel leaping between parallel branches in a dense forest.
10. A tracking shot following a line of elephants walking across a vast, dusty plain.

---

### 场景 3：非人生物非刚性运动 (Non-rigid Motion, Non-human Bio)
*   **要求**：相机位置改变，对象为非人生物，强调肢体剧烈形变。
*   **审计重点**：$\sigma(R_{integrity})$ 结构协同性（检测拉面化/溶解）。

1. A handheld camera records a large octopus swimming away in a complex coral reef, tentacles waving.
2. A moving camera follows a kangaroo hopping rapidly across a rocky outback terrain.
3. A low-angle shot of a large frog leaping toward a pond, camera tracking its jump.
4. A camera moving backward following a snake slithering through dense, colorful flowers on the ground.
5. A handheld shot of a monkey swinging between ropes and moving away into thick jungle foliage.
6. A moving camera follows a peacock walking and shaking its tail feathers in a grand palace garden.
7. A drone shot following a school of fish darting in different directions in a clear blue ocean.
8. A handheld camera follows a baby elephant playing and rolling on the ground while moving away.
9. A moving camera follows a large butterfly fluttering irregularly through a crowded flower market.
10. A tracking shot of a wolf running through a snowy forest, its fur and muscles moving intensely.

---

### 场景 4：环绕旋转场景 (Orbital Rotation, Complex Environment)
*   **要求**：物体在中央，画面围绕其旋转，背景复杂。
*   **审计重点**：自适应修正逻辑（检测旋转中的 2D 投影伪影）。

1. A vintage steam engine in a busy museum; the camera orbits 180 degrees around it, revealing crowds and exhibits.
2. A large stone fountain in a bustling European plaza; camera rotates around it, showing tourists and old buildings.
3. A futuristic car prototype in a high-tech lab; camera circles it, showing flashing monitors and engineers.
4. A complex bronze statue in a city park; camera performs a smooth 360-degree orbit, showing moving traffic behind.
5. A large, ornate clock tower in a city center; camera rotates around its base, capturing the surrounding street life.
6. A detailed spaceship model in a cluttered workshop; camera orbits it, showing tools and blueprints in the background.
7. A massive grand piano on a stage; camera rotates around it, revealing the vast, dimly lit concert hall.
8. A classic carousel in a crowded amusement park; camera orbits the central pillar while it's in motion.
9. A large telescope in an observatory; camera circles it, showing the complex dome structure and control panels.
10. A giant bell in an ancient temple; camera rotates around it, capturing the intricate carvings and monks walking by.

---

### 场景 5：短时部分遮挡 (Partial Occlusion, <1s)
*   **要求**：相机移动，遮挡时间 <1s，物体不被完全遮挡。
*   **审计重点**：遮挡常驻性与尺度跳变检测。

1. A car driving along a street, its wheels briefly obscured by a low roadside guardrail for less than a second.
2. A person walking away, their lower body partially hidden behind a series of thin lamp posts.
3. A train passing behind a row of thin vertical power line poles, camera tracking its movement.
4. A cyclist moving away, partially obscured by the gaps in a picket fence while the camera moves.
5. A cat walking behind a park bench, its head always visible above the slats, camera tracking.
6. A bus moving through a city, partially hidden by a thin traffic sign for a brief moment.
7. A runner in a stadium, seen through the vertical bars of a railing for a split second.
8. A classic car driving past a row of thin trees, never fully disappearing from the camera's view.
9. A boat sailing behind a thin pier support, maintaining partial visibility throughout.
10. A robot walking in a warehouse, passing behind a thin metal rack, camera following.

---

### 场景 6：标准希区柯克变焦 (Dolly Zoom)
*   **要求**：经典的内参/外参耦合挑战。
*   **审计重点**：$f \cdot H / Z$ 联动逻辑。

1. A dramatic dolly zoom on a lone tree in a vast field, the horizon rushing in while the tree stays constant.
2. A cinematic dolly zoom on a person standing at the end of a long, dark library aisle.
3. A vertigo-effect dolly zoom on a statue in a grand hall, the pillars behind it warping.
4. A fast dolly zoom on a red mailbox in a quiet street, the background expanding rapidly.
5. A slow, haunting dolly zoom on an old telephone booth in the middle of a foggy forest.
6. A professional dolly zoom on a lighthouse by the sea, the waves stretching in the distance.
7. A creative dolly zoom on a mountain peak, the valley below appearing to shrink.
8. A dolly zoom on a medieval castle gate, the surrounding mountains growing larger.
9. A dolly zoom on a person looking down a long, symmetrical skywalk.
10. A classic dolly zoom on a doorway at the end of a corridor, the frame distorting intensely.

---

### 🚀 实验执行建议：
1.  **视频时长**：建议控制在 5-10 秒（150-300 帧），以确保位移足够触发表型。
2.  **相机移动**：在 Prompt 中明确 "Handheld" 或 "Moving camera"，这能强制 Mega-SAM 获得必要的视差。
3.  **结果比对**：在场景 5 中，重点检查 PDI 曲线在“遮挡瞬间”是否有尖峰；在场景 4 中，检查 $VP_{fg}$ 是否保持稳定。