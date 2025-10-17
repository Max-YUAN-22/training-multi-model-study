# training-multi-model-study

 ---
  🎯 傻瓜式操作流程（跟着做就行）

  第一步：打开终端 (10秒)

  Ubuntu系统打开终端的方法：
  - 方法1：按键盘 Ctrl + Alt + T
  - 方法2：点左上角"Activities"（活动），搜索"terminal"
  - 方法3：右键桌面，选"Open Terminal"

  你会看到一个黑色或紫色窗口，类似这样：
  ubuntu-user@WS7-3:~$

  ---
  第二步：插入移动硬盘并找到它 (1分钟)

  # 1. 插入你的移动硬盘（Seagate那个）

  # 2. 在终端输入这个命令，查看硬盘挂载在哪里
  lsblk

  # 你会看到类似这样的输出：
  # NAME   SIZE  MOUNTPOINT
  # sda    2TB   /media/ubuntu-user/Seagate
  # 或
  # sdb    2TB   /media/ubuntu-user/Seagate

  # 3. 进入硬盘目录（根据上面的实际路径）
  cd /media/ubuntu-user/Seagate

  # 4. 看看文件在不在
  ls -lh AFMAS_GastricCancer_Dataset
  ls -lh External_Validation_Datasets

⏺ 如果看到文件列表，说明找到了！继续下一步 ✅

  ---
  第三步：复制数据到本地 (5分钟)

  # 1. 创建工作目录
  mkdir -p ~/workspace
  cd ~/workspace

  # 2. 复制数据（这里会花几分钟，等进度条走完）
  echo "开始复制数据..."

  # 复制核心代码
  cp -r /media/ubuntu-user/Seagate/AFMAS_GastricCancer_Dataset ~/workspace/

  # 复制数据集
  cp -r /media/ubuntu-user/Seagate/External_Validation_Datasets ~/workspace/

  echo "✅ 数据复制完成！"

  # 3. 验证复制是否成功
  ls ~/workspace/
  # 你应该看到:
  # AFMAS_GastricCancer_Dataset
  # External_Validation_Datasets

  看到这两个文件夹就OK了！ ✅

  ---
  第四步：安装Python环境 (3分钟)

  # 1. 检查conda是否已安装
  conda --version

  # 如果显示版本号（如 conda 23.x.x），直接跳到步骤2
  # 如果提示"command not found"，运行下面的安装命令：

  # 安装Miniconda（如果没有conda）
  cd ~
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
  ~/miniconda3/bin/conda init bash
  source ~/.bashrc

  # 2. 创建Python环境
  conda create -n afmas python=3.10 -y

  # 3. 激活环境
  conda activate afmas

  # 你会看到命令行前面变成: (afmas) ubuntu-user@WS7-3:~$

  ---
  第五步：安装PyTorch和依赖 (2分钟)

  # 确保在afmas环境中（看到 (afmas) 前缀）
  conda activate afmas

  # 安装PyTorch (支持RTX 4090的CUDA版本)
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

  # 安装其他库
  pip install tqdm Pillow numpy matplotlib scikit-learn

  echo "✅ 安装完成！"

  ---
  第六步：验证GPU可用 (30秒)

  # 检查CUDA是否可用（关键步骤！）
  python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: 
  {torch.cuda.get_device_name(0)}')"

  期望输出：
  CUDA可用: True
  GPU: NVIDIA GeForce RTX 4090

  如果看到这个，恭喜你，环境配置完成！ 🎉

  ---
  第七步：快速测试 (5分钟)

  # 进入代码目录
  cd ~/workspace/AFMAS_GastricCancer_Dataset

  # 给脚本执行权限
  chmod +x quick_start_rtx4090.sh

  # 修改数据路径（自动替换）
  sed -i 's|DATA_DIR=".*"|DATA_DIR="/home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_full/GasH
  isSDB/160"|g' quick_start_rtx4090.sh

  # 运行快速测试
  ./quick_start_rtx4090.sh

  # 当出现选项时，输入数字: 4
  # 按回车

  你会看到训练开始：
  ========================================
        SimCLR RTX 4090 快速测试
  ========================================
  Loading images from: /home/ubuntu-user/workspace/...
  Found 1000 images
  Epoch 1/10 - Loss: 4.2341 - Time: 35s
  Epoch 2/10 - Loss: 3.8923 - Time: 32s
  ...

  如果看到这些，测试成功！ ✅

  ---
  第八步：启动正式训练 (自动运行1-1.5小时)

  测试成功后，按Ctrl+C停止测试，然后运行：

  # 后台启动完整训练（100 epochs）
  nohup python3 27_simclr_pretraining_cuda.py \
      --data_dir /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_full/GasHisSDB/160 \
      --num_epochs 100 \
      --batch_size 256 \
      --num_workers 8 \
      --use_amp \
      --log_dir ./simclr_logs_rtx4090 \
      > ~/simclr_training.log 2>&1 &

  echo "✅ 训练已启动！进程ID: $!"

  # 实时查看训练进度
  tail -f ~/simclr_training.log

  # 按 Ctrl+C 可以退出查看（训练会继续在后台运行）

  ---
  第九步：监控GPU (随时查看)

  打开一个新的终端窗口（Ctrl+Alt+T），运行：

  # 实时GPU监控
  watch -n 1 nvidia-smi

  # 你会看到GPU使用率95%以上，显存18-22GB

  ---
  📋 完整复制粘贴版本（懒人专用）

  如果你想一次性复制所有命令，用这个：

  # ==================== 完整流程 ====================

  # 1. 找到硬盘
  lsblk
  cd /media/ubuntu-user/Seagate  # 根据实际路径调整

  # 2. 复制数据
  mkdir -p ~/workspace
  cp -r /media/ubuntu-user/Seagate/AFMAS_GastricCancer_Dataset ~/workspace/
  cp -r /media/ubuntu-user/Seagate/External_Validation_Datasets ~/workspace/

  # 3. 安装环境（如果没有conda，先装miniconda）
  conda create -n afmas python=3.10 -y
  conda activate afmas

  # 4. 安装PyTorch
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install tqdm Pillow numpy matplotlib scikit-learn

  # 5. 验证CUDA
  python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

  # 6. 启动训练
  cd ~/workspace/AFMAS_GastricCancer_Dataset
  nohup python3 27_simclr_pretraining_cuda.py \
      --data_dir /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_full/GasHisSDB/160 \
      --num_epochs 100 \
      --batch_size 256 \
      --num_workers 8 \
      --use_amp \
      --log_dir ./simclr_logs_rtx4090 \
      > ~/simclr_training.log 2>&1 &

  echo "训练进程ID: $!"

  # 7. 查看进度
  tail -f ~/simclr_training.log

  ---
  ⏱️ 时间表

  | 步骤       | 时间      | 说明              |
  |----------|---------|-----------------|
  | 1. 打开终端  | 10秒     | Ctrl+Alt+T      |
  | 2. 找到硬盘  | 1分钟     | lsblk           |
  | 3. 复制数据  | 5分钟     | 取决于USB速度        |
  | 4-5. 装环境 | 5分钟     | conda + PyTorch |
  | 6. 验证GPU | 30秒     | 测试CUDA          |
  | 7. 快速测试  | 5分钟     | 10 epochs测试     |
  | 8. 正式训练  | 1-1.5小时 | 自动后台运行          |

  总计：15分钟手动操作 + 1小时自动训练

  ---
  🆘 遇到问题怎么办？

  问题1：找不到硬盘
  # 重新插拔硬盘，然后
  lsblk
  df -h | grep media

  问题2：conda命令不存在
  # 安装miniconda
  cd ~
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
  ~/miniconda3/bin/conda init bash
  source ~/.bashrc

  问题3：CUDA不可用
  # 检查GPU驱动
  nvidia-smi

  # 重装PyTorch
  pip3 uninstall torch torchvision torchaudio
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

  ---
  现在开始吧！

  1. 打开终端（Ctrl+Alt+T）
  2. 插入移动硬盘
  3. 复制上面的命令，一步步粘贴执行

  遇到任何问题立即截图告诉我！ 🚀
