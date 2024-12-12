1. 项目背景
在现代人机交互中，手势识别作为一种自然直观的交互方式，得到广泛关注和应用。对于人形机器人来说，能够准确识别和理解用户的手势指令，是实现流畅互动、成功执行复杂任务的关键。本项目期望开发一套基于普通RGB摄像头的动态手势识别系统，帮助人形机器人实时捕捉并识别用户的手势指令（如挥手、握手），从而实现更智能和人性化的互动。
2. 项目要求
2.1 实时性要求
- 帧率 (FPS) ≥ 20 FPS：确保手势识别过程的实时性，避免因帧率过低导致的识别延迟和交互不流畅。
2.2 动态手势识别
- 识别动态手势：系统需识别诸如挥手、握手等需要手部运动的手势，确保能够捕捉手势的运动轨迹。
- 运动轨迹分析：通过分析手腕位置的历史变化，判断手势的动态特征，提高识别的准确性。
2.3 左右手识别
- 区分左右手：通过MediaPipe提供的手部识别功能，准确区分用户的左右手，并分别进行手势识别。
- 独立识别双手手势：支持用户同时使用双手进行不同手势操作，系统能够独立识别并标注每只手的手势。
2.4 手势种类及可扩展性
- 支持多种基本手势：当前系统支持如挥手、握手、比心、点赞、OK等手势的识别。
- 可扩展的手势库：系统设计允许开发者方便地添加新的手势识别逻辑，扩展手势库以满足更多应用场景需求。
2.5 本地部署与硬件要求
- 本地运行：无需依赖云端服务，系统可在普通台式机或笔记本电脑上运行。
- 摄像头兼容性：支持多种型号的RGB摄像头，用户可使用内置或外接摄像头进行手势捕捉。
2.6 准确性与容错性
- 高识别准确率：通过精确的手部关键点检测和角度计算，确保手势识别的高准确性。
- 环境适应性：系统在不同光照条件、复杂背景和用户不同姿态下，仍能稳定运行，具备一定的容错能力。
3. 系统概述
本系统利用普通RGB摄像头实时捕捉用户的手部图像，借助MediaPipe Hands模块进行手部关键点的检测与追踪。通过计算手指关节的角度、手掌方向以及手部的运动轨迹，系统能够实时识别多种动态手势。识别结果会在视频帧中实时显示，并可进一步传递给机器人系统，以触发相应的动作或反应。
4. 功能描述
4.1 实时手势识别
系统能够实时捕捉视频流中的手部信息，分析并识别多种动态手势。以下表格列出了系统目前支持的手势、其对应的含义以及识别方法。
暂时无法在飞书文档外展示此内容
4.2 手势识别逻辑
1. 手部关键点检测：
  - 使用MediaPipe Hands模块检测手部21个关键点的二维和三维坐标。
2. 角度计算：
  - 通过计算手指关节的二维和三维角度，评估每个手指的弯曲状态。
3. 手掌方向分析：
  - 计算手掌法线向量与摄像头方向的夹角，判断手掌的朝向。
4. 动态手势分析：
  - 通过历史手腕位置的变化，识别具有运动特征的手势（如挥手）。
5. 手势分类：
  - 根据手指的弯曲状态、手掌方向和运动轨迹，将手势分类为预定义的类别。
5. 技术实现
5.1 使用的技术与库
- OpenCV：用于视频捕捉、图像处理和显示。
- MediaPipe：用于手部关键点检测和跟踪。
- NumPy：用于数值计算和向量运算。
5.2 手势识别算法
5.2.1 手指角度计算
- 二维角度：计算指根与指尖的方向向量之间的夹角，判断手指是否弯曲或伸直。
- 三维角度：通过三维向量计算手掌法线与摄像头方向的夹角，判断手掌的朝向。
5.2.2 手掌法线计算
- 使用手腕、食指根部和小指根部的三维坐标，计算手掌的法线向量，进而判断手掌与摄像头的相对方向。
5.2.3 手势分类逻辑
- 手指状态判定：根据计算得到的角度列表，判断每个手指是伸直还是弯曲。
- 手掌方向判定：通过手掌法线与摄像头方向的夹角，判断手掌的朝向。
- 动态动作判定：利用手腕的历史位置变化，识别具有运动特征的手势（如挥手）。
- 手势匹配：将手指状态、手掌方向和动态特征与预定义的手势模板进行匹配，确定当前手势。
5.3 代码结构概述
- vector_2d_angle(v1, v2)：计算两个二维向量之间的夹角（0~180度）。用于判断手指的弯曲状态。
- vector_3d_angle(v1, v2)：计算两个三维向量之间的夹角（0~180度）。用于判断手掌法线与摄像头方向的夹角。
- hand_angle(hand_landmarks)：计算手指关节的角度列表，返回[拇指, 食指, 中指, 无名指, 小拇指]各自的角度。
- h_gesture(angle_list, palm_angle, palm_facing_angle, fingers_pointing_toward_screen, is_all_fingers_open, history_x, hand_local, normal, move_thr=30)：根据角度列表、手掌方向、手指朝向、手腕历史位置等信息，判断手势名称。
- detect()：主函数，负责摄像头初始化、手部检测、手势识别和结果显示。按 'q' 键退出程序。
6. 用户界面
- 视频窗口：实时显示摄像头捕捉到的视频流，绘制手部关键点和连接线。
- 手势标注：在视频帧中显示识别到的手势名称，如“Wave”、“Fist”等，标注在手腕位置上方。
- FPS显示：在视频帧的角落实时显示当前的帧率，帮助用户了解系统的实时性表现。
7. 部署说明
7.1 环境要求
- 操作系统：Windows、macOS 或 Linux
- Python 版本：3.7及以上
- 硬件要求：普通计算机，内置或外接RGB摄像头
- 依赖库：
  - OpenCV
  - MediaPipe
  - NumPy
7.2 安装步骤
1. 安装Python：
  - 确保系统已安装Python 3.7及以上版本。可从Python官网下载并安装。
2. 安装依赖库：
  - 打开终端或命令提示符，运行以下命令安装所需库：
pip install opencv-python mediapipe numpy
3. 下载项目代码：
  - 获取项目代码文件，确保所有代码文件（如detect.py）位于同一目录下。
4. 运行程序：
  - 在终端或命令提示符中导航到项目目录，运行以下命令启动程序：
python detect.py
  - 程序启动后，会打开一个名为“Hand Gesture Recognition”的窗口，开始实时手势识别。
7.3 使用说明
启动程序：运行detect.py脚本后，系统将自动打开摄像头并开始识别手势。
- 手势识别：将手部置于摄像头前，做出预定义的手势（如挥手、握手等），系统会实时识别并在视频窗口中显示手势名称。
- 退出程序：按下键盘上的'q'键，程序将退出并关闭所有窗口。
8. 测试与验证
8.1 功能测试
- 单手识别：测试每个预定义手势，确保系统能够准确识别单只手的手势。
- 双手识别：同时展示左右手不同手势，验证系统能分别识别左右手的手势。
- 动态手势：测试挥手等动态手势，确保系统能识别手部的运动轨迹。
8.2 性能测试
- 帧率测试：在不同计算机配置下测试系统的FPS，确保在各种硬件环境下都能达到不低于20 FPS的要求。
- 环境适应性测试：在不同光照、背景和用户姿态下测试系统的识别准确性和稳定性，确保系统具备良好的环境适应性。
8.3 容错性测试
- 误识别测试：输入非预定义手势，确保系统不会误识别为其他手势，验证系统的误识别率。
- 部分遮挡测试：手部部分被遮挡或光线不足时，验证系统的容错能力和识别稳定性。
9. 未来增强
- 手势扩展：增加更多复杂和多样化的手势识别，如数字手势、字母手势等，以满足更多应用场景需求。
- 多模态识别：结合语音识别或其他传感器数据，提升交互的丰富性和准确性，实现更加智能的人机互动。
- 优化算法：进一步优化手势识别算法，提升识别速度和准确率，降低对硬件的依赖，确保在更广泛的设备上稳定运行。
- 用户自定义手势：允许用户自定义手势，通过简单的训练过程让系统识别新的手势，增强系统的个性化和灵活性。
- 跨平台支持：扩展系统的兼容性，支持更多操作系统和设备平台，提升系统的普及性和应用范围。
10. 结论
本项目通过结合OpenCV和MediaPipe技术，实现了一个高效、准确的动态手势实时识别系统。系统能够满足实时性、准确性和可扩展性的基本要求，支持多种手势的识别，并具备左右手的独立识别能力。系统设计简洁，易于在普通计算机上部署和运行，为人形机器人的互动提供了可靠的手势识别支持。未来将继续优化算法，扩展手势库，提升系统的应用范围和用户体验。
附录
A. 手势识别表格详解
暂时无法在飞书文档外展示此内容
B. 参数说明
- thr_angle：用于判定手指是否弯曲的角度阈值（默认65度）。角度大于此值的手指被认为是弯曲的。
- thr_angle_s：用于判定手指是否伸直的角度阈值（默认49度）。角度小于此值的手指被认为是伸直的。
- move_thr：用于Wave手势检测的手腕x坐标移动阈值（默认20像素）。当手腕x坐标的变化幅度超过此阈值时，判定为Wave手势。
- gesture_history：记录手腕x坐标的历史，用于检测水平移动（如Wave手势的判定）。
C. 主要函数说明
1. vector_2d_angle(v1, v2)
计算两个二维向量之间的夹角（0~180度）。用于判断手指的弯曲状态。
- 参数：
  - v1：第一个二维向量，格式为(x, y)。
  - v2：第二个二维向量，格式为(x, y)。
- 返回值：
  - 夹角（浮点数），范围0~180度。若计算异常则返回65535。
2. vector_3d_angle(v1, v2)
计算两个三维向量之间的夹角（0~180度）。用于判断手掌法线与摄像头方向的夹角。
- 参数：
  - v1：第一个三维向量，格式为[x, y, z]。
  - v2：第二个三维向量，格式为[x, y, z]。
- 返回值：
  - 夹角（浮点数），范围0~180度。若计算异常则返回65535。
3. hand_angle(hand_landmarks)
计算手指关节的角度列表，返回[拇指, 食指, 中指, 无名指, 小拇指]各自的角度。
- 参数：
  - hand_landmarks：手部关键点的二维坐标列表，格式为[(x1, y1), (x2, y2), ..., (x21, y21)]。
- 返回值：
  - 手指角度列表，包含五个浮点数，分别对应拇指、食指、中指、无名指、小拇指的角度。
4. h_gesture(angle_list, palm_angle, palm_facing_angle, fingers_pointing_toward_screen, is_all_fingers_open, history_x, hand_local, normal, move_thr=30)
根据角度列表、手掌方向、手指朝向、历史位置等信息，判断手势名称。
- 参数：
  - angle_list：手指角度列表。
  - palm_angle：手掌角度（平面内）。
  - palm_facing_angle：手掌法线与摄像头方向夹角。
  - fingers_pointing_toward_screen：手指是否朝向屏幕（通过z值判断）。
  - is_all_fingers_open：是否所有手指打开。
  - history_x：手腕x坐标的历史记录。
  - hand_local：当前手的21个关键点的三维坐标。
  - normal：手掌法线向量。
  - move_thr：Wave手势检测的移动阈值。
- 返回值：
  - gesture_str：识别出的手势名称字符串。
5. detect()
主函数，负责摄像头初始化、手部检测、手势识别和结果显示。按 'q' 键退出程序。
- 功能：
  - 初始化摄像头并设置分辨率。
  - 使用MediaPipe Hands模块进行手部关键点检测和追踪。
  - 计算手指角度、手掌方向和运动轨迹。
  - 识别并标注手势名称和当前帧率（FPS）。
  - 实时显示视频流和识别结果，支持按 'q' 键退出程序。
结束语
通过本功能文档的详细描述，开发人员和使用者可以全面了解动态手势实时识别系统的功能、实现方法和使用方式。系统的模块化设计和清晰的手势识别逻辑为后续的功能扩展和优化提供了坚实的基础。该系统在普通计算机和标准RGB摄像头下即可运行，具备较高的实时性和准确性，适用于人形机器人等需要自然手势交互的应用场景。
版权声明
本项目文档及代码均基于开源技术开发，遵循相关开源协议。未经许可，不得用于商业用途或分发。如需进一步了解或使用，欢迎联系项目维护者。
联系方式
如有任何问题或建议，请联系项目维护者：
- 邮箱：
- GitHub：https://github.com/your-repo
