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









try1.0



  (base) ubuntu-user@WS7-3:~$ ^C
(base) ubuntu-user@WS7-3:~$ cd /media/ubuntu-user/Seagate
(base) ubuntu-user@WS7-3:/media/ubuntu-user/Seagate$ ls -lh AFMAS_GastricCancer_Dataset
total 13M
-rwxr-xr-x 1 ubuntu-user ubuntu-user  14K Oct 14 20:22 01_deduplicate_and_organize.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  12K Oct 14 20:25 02_three_phase_controller.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  14K Oct 14 20:26 03_covariance_collaboration.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  15K Oct 14 20:37 04_agent_base.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  13K Oct 14 20:32 05_agent_models.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  13K Oct 14 20:33 06_diversity_maintenance.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  15K Oct 14 20:56 07_afmas_system.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user 9.4K Oct 14 20:46 08_integrate_xgboost.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  12K Oct 14 20:57 09_integrate_xgboost_vgg.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user 3.7K Oct 14 20:53 10_create_feature_selector.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user 1.7K Oct 14 20:56 11_recreate_selector.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  13K Oct 14 21:04 12_test_on_real_data.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  19K Oct 15 02:00 13_train_agents.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  13K Oct 15 08:04 13_train_agents_resume.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  14K Oct 14 21:21 14_run_experiments.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  15K Oct 14 22:02 15_attention_visualization.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user 9.2K Oct 15 20:06 15b_simple_gradcam.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  16K Oct 15 20:03 15_gradcam_visualization.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  21K Oct 15 21:47 16_afmas_v2_optimized.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  16K Oct 15 01:35 16_statistical_tests.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  15K Oct 15 22:00 17_comparison_analysis.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  15K Oct 15 01:36 17_generate_paper_tables.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  21K Oct 15 11:20 18_generate_sci_quality_figures.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  16K Oct 15 22:09 18_overfitting_analysis.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  15K Oct 15 22:28 19_publication_readiness_assessment.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  18K Oct 15 17:20 19_train_ensemble_agent.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  12K Oct 16 15:58 23_domain_discriminator.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  14K Oct 16 16:21 24_gradient_reversal_layer.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  22K Oct 16 16:30 25_da_afmas.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user 5.9K Oct 16 16:05 26_sample_labeled_data.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  14K Oct 16 16:42 27_simclr_pretraining_cuda.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  15K Oct 16 16:28 27_simclr_pretraining.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user 1.5K Oct 15 21:58 afmas_v2_results.json
-rwxr-xr-x 1 ubuntu-user ubuntu-user 264K Oct 14 20:26 agent_correlation_matrix.png
-rwxr-xr-x 1 ubuntu-user ubuntu-user 309K Oct 15 13:12 auto_experiments.log
-rwxr-xr-x 1 ubuntu-user ubuntu-user 7.0K Oct 15 11:20 auto_run_all_experiments.sh
-rwxr-xr-x 1 ubuntu-user ubuntu-user 3.4K Oct 15 01:38 check_training_status.sh
drwxr-xr-x 2 ubuntu-user ubuntu-user 128K Oct 15 22:00 comparison_results
-rwxr-xr-x 1 ubuntu-user ubuntu-user 7.4K Oct 15 20:10 CRITICAL_DIAGNOSIS_REPORT.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user  846 Oct 14 20:23 dataset_report.json
-rwxr-xr-x 1 ubuntu-user ubuntu-user 328K Oct 14 20:33 diversity_history.png
drwxr-xr-x 2 ubuntu-user ubuntu-user 128K Oct 15 05:40 experiment_results
-rwxr-xr-x 1 ubuntu-user ubuntu-user 155K Oct 15 20:05 experiment_run.log
drwxr-xr-x 2 ubuntu-user ubuntu-user 128K Oct 14 20:08 experiments
-rwxr-xr-x 1 ubuntu-user ubuntu-user 278K Oct 15 16:46 experiments_full.log
-rwxr-xr-x 1 ubuntu-user ubuntu-user  19K Oct 15 15:26 experiments.log
-rwxr-xr-x 1 ubuntu-user ubuntu-user 2.4K Oct 15 13:12 EXPERIMENT_SUMMARY.md
drwxr-xr-x 2 ubuntu-user ubuntu-user 128K Oct 16 00:51 external_validation
-rwxr-xr-x 1 ubuntu-user ubuntu-user 6.4K Oct 15 23:14 external_validation_options.md
drwxr-xr-x 6 ubuntu-user ubuntu-user 128K Oct 15 20:03 gradcam_results
drwxr-xr-x 4 ubuntu-user ubuntu-user 128K Oct 14 20:08 models
-rwxr-xr-x 1 ubuntu-user ubuntu-user 8.8K Oct 14 21:27 NEXT_STEPS.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user  903 Oct 15 22:09 overfitting_analysis_report.json
-rwxr-xr-x 1 ubuntu-user ubuntu-user  19K Oct 14 21:21 PAPER_OUTLINE.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user  13K Oct 15 01:37 PAPER_WRITING_GUIDE.md
drwxr-xr-x 5 ubuntu-user ubuntu-user 128K Oct 14 20:22 processed_data
-rwxr-xr-x 1 ubuntu-user ubuntu-user  13K Oct 14 21:08 PROJECT_STATUS_FINAL.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user  16K Oct 14 20:41 PROJECT_SUMMARY.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user  10K Oct 15 17:25 PUBLICATION_ROADMAP.md
drwxr-xr-x 2 ubuntu-user ubuntu-user 128K Oct 14 20:36 __pycache__
-rwxr-xr-x 1 ubuntu-user ubuntu-user 4.2K Oct 14 21:26 quick_commands.sh
-rwxr-xr-x 1 ubuntu-user ubuntu-user 4.8K Oct 16 16:45 quick_start_rtx4090.sh
drwxr-xr-x 2 ubuntu-user ubuntu-user 128K Oct 14 20:08 raw_data
-rwxr-xr-x 1 ubuntu-user ubuntu-user 8.1K Oct 14 20:41 README.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user 9.8K Oct 14 21:23 README_PROJECT.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user 9.7K Oct 14 21:06 REAL_DATA_TEST_REPORT.md
drwxr-xr-x 3 ubuntu-user ubuntu-user 128K Oct 14 20:08 results
-rwxr-xr-x 1 ubuntu-user ubuntu-user 8.1K Oct 16 16:45 RTX4090_DEPLOYMENT_GUIDE.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user 8.3K Oct 14 21:26 STATUS_REPORT.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user 2.8K Oct 14 21:05 test_results_real_data.json
-rwxr-xr-x 1 ubuntu-user ubuntu-user 692K Oct 15 11:51 training_log_gpu_resume.txt
-rwxr-xr-x 1 ubuntu-user ubuntu-user 991K Oct 15 05:38 training_log_gpu.txt
-rwxr-xr-x 1 ubuntu-user ubuntu-user  75K Oct 15 02:08 training_log.txt
-rwxr-xr-x 1 ubuntu-user ubuntu-user 6.7K Oct 16 16:47 TRANSFER_TO_LAB.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user 8.4K Oct 14 20:59 XGBOOST_INTEGRATION_COMPLETE.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user 1.7K Oct 14 20:57 xgboost_selector.py
(base) ubuntu-user@WS7-3:/media/ubuntu-user/Seagate$ ls -lh External_Validation_Datasets
total 2.7M
-rwxr-xr-x 1 ubuntu-user ubuntu-user  13K Oct 16 00:36 20_external_validation_gashissdb.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user 9.0K Oct 16 14:18 21_external_validation_simple.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  19K Oct 16 10:14 22_analyze_validation_results.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  13K Oct 16 00:49 agent_models_05.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user 6.8K Oct 16 00:38 CURRENT_STATUS_SUMMARY.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user 3.5K Oct 16 00:22 download_gashissdb.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user  11K Oct 16 00:46 EXECUTIVE_SUMMARY.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user 3.5K Oct 16 10:19 extract_gashissdb.py
-rwxr-xr-x 1 ubuntu-user ubuntu-user 8.2K Oct 16 11:00 FILES_CREATED_SUMMARY.md
drwxr-xr-x 6 ubuntu-user ubuntu-user 128K Oct 16 00:18 GasHisSDB
drwxr-xr-x 4 ubuntu-user ubuntu-user 128K Oct 16 00:22 GasHisSDB_full
-rwxr-xr-x 1 ubuntu-user ubuntu-user 5.2K Oct 16 00:35 GasHisSDB_INFO.md
drwxr-xr-x 4 ubuntu-user ubuntu-user 128K Oct 16 16:08 GasHisSDB_labeled_1k
-rwxr-xr-x 1 ubuntu-user ubuntu-user  11K Oct 16 10:54 NEXT_STEPS_GUIDE.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user 9.6K Oct 16 10:05 PAPER_EXTERNAL_VALIDATION_SECTION.md
drwxr-xr-x 2 ubuntu-user ubuntu-user 128K Oct 16 00:49 __pycache__
-rwxr-xr-x 1 ubuntu-user ubuntu-user 6.2K Oct 16 10:59 README.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user 9.2K Oct 16 10:17 REFERENCES_CITATIONS.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user  40K Oct 16 14:54 validation_gpu.log
-rwxr-xr-x 1 ubuntu-user ubuntu-user  11K Oct 16 10:57 WORK_COMPLETED_OVERNIGHT.md
-rwxr-xr-x 1 ubuntu-user ubuntu-user  11K Oct 16 00:53 WORK_COMPLETED_SUMMARY.md
(base) ubuntu-user@WS7-3:/media/ubuntu-user/Seagate$ mkdir -p ~/workspace
(base) ubuntu-user@WS7-3:/media/ubuntu-user/Seagate$ cd ~/workspace
(base) ubuntu-user@WS7-3:~/workspace$ echo "开始复制数据..."

开始复制数据...
(base) ubuntu-user@WS7-3:~/workspace$ 



修复try1

cp -rv /media/ubuntu-user/Seagate/AFMAS_GastricCancer_Dataset ~/workspace/
cp -rv /media/ubuntu-user/Seagate/External_Validation_Datasets ~/workspace/
echo "✅ 数据复制完成！"



try2


cd ~/workspace/AFMAS_GastricCancer_Dataset

  nohup python3 27_simclr_pretraining_cuda.py \
      --data_dir /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_full/GasHisSDB/160 \
      --num_epochs 100 \
      --batch_size 256 \
      --num_workers 8 \
      --use_amp \
      --log_dir ./simclr_logs_rtx4090 \
      > ~/simclr_full_training.log 2>&1 &

  echo "✅ 完整训练已启动！进程ID: $!"
  echo $! > ~/simclr_training.pid




  try3

  # 1. 停止错误的进程
  pkill -f 27_simclr_pretraining_cuda.py

  # 2. 激活环境
  conda activate afmas

  # 3. 进入目录
  cd ~/workspace/AFMAS_GastricCancer_Dataset

  # 4. 单行启动命令（一次性复制这整行）
  nohup python3 27_simclr_pretraining_cuda.py --data_dir /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_full/GasHisSDB/160 --num_epochs
   100 --batch_size 256 --num_workers 8 --use_amp --log_dir ./simclr_logs_rtx4090 > ~/simclr_full_training.log 2>&1 &

  # 5. 记录进程ID
  echo "训练进程ID: $!" | tee ~/simclr_training.pid

  # 6. 等待2秒后查看是否启动成功
  sleep 2
  tail -n 15 ~/simclr_full_training.log

  把这整个代码块复制粘贴到终端！ 这样可以确保命令在一行执行。

  执行后你应该看到类似：
  [1] 12345
  训练进程ID: 12345
  ================================================================================
                          🚀 SimCLR CUDA Optimized Training
  ================================================================================
  ✓ 使用设备: NVIDIA GeForce RTX 4090
