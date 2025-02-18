
---

# 高效人物跟踪与识别  
**Efficient Person Tracking and Identification**

# 项目展示
**Demonstration**

![项目演示](output/tracked_video.gif)
![项目演示2](output/tracked_video_2.gif)

## 项目概述  
**Project Overview**  
本项目结合了 **DeepSORT** 和 **TorchReID** 实现了高效的多人物跟踪与身份识别系统，特别适用于复杂环境中，如遮挡严重或跟踪丢失后的快速恢复身份。通过整合 **YOLOv11** 进行目标检测和 **DeepSORT** 进行多帧目标跟踪，本系统可以高效实时追踪运动场上的运动员，并保证身份一致性。  
This project integrates **DeepSORT** and **TorchReID** to create an efficient multi-person tracking and identification system, especially suited for complex environments, such as when occlusions are severe or identity needs to be quickly recovered after tracking loss. By combining **YOLOv11** for object detection and **DeepSORT** for multi-frame tracking, the system can track athletes on a sports field in real-time, ensuring identity consistency.

## 核心功能  
**Core Features**  
- **YOLOv11**：进行实时目标检测，快速识别和定位运动员。  
  **YOLOv11**: Real-time object detection for quick identification and localization of athletes.  
- **DeepSORT**：高效的目标跟踪，确保每个运动员的连续跟踪，并提供唯一身份ID。  
  **DeepSORT**: Efficient tracking of targets to ensure continuous tracking of each athlete, providing a unique identity ID.  
- **TorchReID**：确保跟踪过程中的身份一致性。  
  **TorchReID**: Ensuring identity consistency throughout the tracking process.  
- **多帧融合跟踪**：增强跟踪的鲁棒性，在复杂环境中保持高准确度，即使在遮挡或目标快速变换位置时也能有效跟踪。  
  **Multi-frame Fusion Tracking**: Enhances tracking robustness, maintaining high accuracy in complex environments, even when occlusions occur or targets move quickly.

## 适用场景  
**Applicable Scenarios**  
- **运动场景**：运动员的实时识别和跟踪，支持快速身份恢复和一致性维护。  
  **Sports Scenarios**: Real-time identification and tracking of athletes, with support for quick identity recovery and consistency maintenance.  
- **复杂环境**：适用于遮挡严重、光线变化等复杂场景下的多目标跟踪和识别。  
  **Complex Environments**: Suitable for multi-target tracking and identification in complex scenarios, such as severe occlusions or lighting changes.  
- **高密度人群**：在人员密集、快速移动的环境下保持高效准确的目标跟踪。  
  **High-Density Crowds**: Maintains efficient and accurate target tracking in environments with dense crowds and fast movement.

## 安装与使用  
**Installation and Usage**

### 环境要求  
**System Requirements**  
- Python 3.x  
- 依赖库：  
  - `YOLO`（用于目标检测）  
  - `DeepSORT`（用于目标跟踪）  
  - `TorchReID`（用于身份识别）  
- Python 3.x  
- Dependencies:  
  - `YOLO` (for object detection)  
  - `DeepSORT` (for object tracking)  
  - `TorchReID` (for identity recognition)

### 安装依赖  
**Install Dependencies**  
```bash  
pip install -r requirements.txt  
```  

--- 
