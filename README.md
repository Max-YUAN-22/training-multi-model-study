  # 注释掉多层级判别器的调用
  sed -i '318,333s/^/#/' 25_da_afmas.py

  # 验证
  sed -n '318,333p' 25_da_afmas.py

  执行后重新启动：
  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data
  /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 >
  ~/da_afmas_training.log 2>&1 &
  sleep 3 && tail -n 40 ~/da_afmas_training.log
  
try

(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 > ~/da_afmas_training.log 2>&1 &

[1] 16463
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sleep 3 && tail -n 40 ~/da_afmas_training.log
[1]+  Exit 1                  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 > ~/da_afmas_training.log 2>&1
Traceback (most recent call last):
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/25_da_afmas.py", line 535, in <module>
    outputs = model(x, domain_label=domain_labels, training=True)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/25_da_afmas.py", line 333, in forward
    multi_level_logits = self.multi_level_discriminator(multi_level_reversed)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py", line 171, in forward
    logits = discriminator(features)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py", line 92, in forward
    domain_logits = self.discriminator(features)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/container.py", line 244, in forward
    input = module(input)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x1536 and 2048x512)


try###

npm install -g @anthropic-ai/claude-code --registry=https://registry.npmmirror.com



nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data
  /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 >
  ~/da_afmas_training.log 2>&1 &

  等待3秒后查看日志：
  sleep 3 && tail -n 40 ~/da_afmas_training.log
  
 try
  (base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ grep -n "use_multi_level" 25_da_afmas.py | head -5

97:        use_multi_level_adaptation: bool = True,
107:            use_multi_level_adaptation: 是否使用多层级域适应
115:        self.use_multi_level = use_multi_level_adaptation
199:        if use_multi_level_adaptation:
318:        if training and self.use_multi_level and agent_features is not None:
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ \sed -i 's/use_multi_level_adaptation: bool = True/use_multi_level_adaptation: bool = False/g' 25_da_afmas.py
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ grep "use_multi_level_adaptation" 25_da_afmas.py | head -3
        use_multi_level_adaptation: bool = False,
            use_multi_level_adaptation: 是否使用多层级域适应
        self.use_multi_level = use_multi_level_adaptation



⏺ 现在是1536维！agent特征实际维度和预期不同。

  最简单的解决方案：禁用多层级判别器，只用单层的：

  # 查看use_multi_level的设置
  grep -n "use_multi_level" 25_da_afmas.py | head -5

  或者直接修改，禁用多层级判别器：

  # 禁用多层级判别器（改为False）
  sed -i 's/use_multi_level_adaptation: bool = True/use_multi_level_adaptation: bool = False/g' 25_da_afmas.py

  # 验证
  grep "use_multi_level_adaptation" 25_da_afmas.py | head -3



(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sed -i 's/feature_dims=(512, 1024, 2048)/feature_dims=(2048, 2048, 2048)/g' 25_da_afmas.py
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ grep "feature_dims=" 25_da_afmas.py
                feature_dims=(2048, 2048, 2048),
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 > ~/da_afmas_training.log 2>&1 &
[1] 15523
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sleep 3 && tail -n 40 ~/da_afmas_training.log
[1]+  Exit 1                  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 > ~/da_afmas_training.log 2>&1
Traceback (most recent call last):
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/25_da_afmas.py", line 535, in <module>
    outputs = model(x, domain_label=domain_labels, training=True)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/25_da_afmas.py", line 333, in forward
    multi_level_logits = self.multi_level_discriminator(multi_level_reversed)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py", line 171, in forward
    logits = discriminator(features)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py", line 92, in forward
    domain_logits = self.discriminator(features)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/container.py", line 244, in forward
    input = module(input)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x1536 and 2048x512)




  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data
  /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 >
  ~/da_afmas_training.log 2>&1 &

  等待3秒后查看日志：
  sleep 3 && tail -n 40 ~/da_afmas_training.log



  
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sed -n '230,250p' 25_da_afmas.py

            return_agent_features: 是否返回各agent的单独特征

        返回:
            combined_features: 融合后的特征 (batch_size, 2048)
            agent_features: 各agent的特征列表（如果return_agent_features=True）
        """
        agent_features = []
        agent_logits = []

        # 从每个agent提取特征
        for agent in self.agents:
            logit, feat = agent(x, return_features=True)
            agent_features.append(feat)
            agent_logits.append(logit)

        # 拼接所有特征
        concatenated_features = torch.cat(agent_features, dim=1)  # (B, total_feat_dim)

        # 特征投影（降维到2048）
        projected_features = self.feature_projector(concatenated_features)  # (B, 2048)

(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sed -i 's/feature_dims=(512, 1024, 2048)/feature_dims=(2048, 2048, 2048)/g' 25_da_afmas.py
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ grep "feature_dims=" 25_da_afmas.py
                feature_dims=(2048, 2048, 2048),
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ 



⏺ 找到了！问题在第322-325行。代码假设：
  - agent_features[0] 是512维
  - agent_features[1] 是1024维
  - features 是2048维

  但实际上所有agent的输出可能都是相同维度的。

  查看agent_features的实际维度：

  sed -n '230,250p' 25_da_afmas.py

  快速修复方案 - 修改MultiLevelDomainDiscriminator的初始化，让所有层都使用2048维：

  # 修复维度不匹配（所有层都用2048）
  sed -i 's/feature_dims=(512, 1024, 2048)/feature_dims=(2048, 2048, 2048)/g' 25_da_afmas.py

  # 验证修复
  grep "feature_dims=" 25_da_afmas.py
  

⏺ 找到了问题！MultiLevelDomainDiscriminator 期望 (512, 1024, 2048) 维度的特征，但实际传入的第一个特征是2048维。

  需要查看在forward方法中如何传递multi_level_features：

  grep -n "multi_level_discriminator" 25_da_afmas.py | grep -v "self.multi_level"

  然后查看具体调用的代码：

  sed -n '320,340p' 25_da_afmas.py


(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ grep -n "MultiLevelDomainDiscriminator" 25_da_afmas.py
37:MultiLevelDomainDiscriminator = domain_discriminator.MultiLevelDomainDiscriminator
201:            self.multi_level_discriminator = MultiLevelDomainDiscriminator(
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sed -n '200,250p' 25_da_afmas.py | grep -A 10 -B 5 "MultiLevel"
            # 假设提取3个层级的特征
            self.multi_level_discriminator = MultiLevelDomainDiscriminator(
                feature_dims=(512, 1024, 2048),
                dropout_rate=dropout_rate
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化新增层的权重"""
        for m in [self.feature_projector, self.task_classifier, self.agent_weight_generator]:
            for layer in m:
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ grep -n "multi_level_discriminator" 25_da_afmas.py | grep -v "self.multi_level"

(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sed -n '320,340p' 25_da_afmas.py
            if len(agent_features) >= 3:
                multi_level_feats = tuple([
                    agent_features[0],  # 512维
                    agent_features[1],  # 1024维
                    features            # 2048维
                ])

                # 应用GRL
                multi_level_reversed = tuple([
                    self.grl(feat) for feat in multi_level_feats
                ])

                # 多层级域判别
                multi_level_logits = self.multi_level_discriminator(multi_level_reversed)
                outputs['multi_level_domain_logits'] = multi_level_logits

        # 6. 可选: 返回特征
        if return_features:
            outputs['features'] = features
            outputs['agent_features'] = agent_features


⏺ 找到了！MultiLevelDomainDiscriminator 的默认 feature_dims=(512, 1024, 2048)，但实际传入的第一个特征是2048维的。

  需要查看25_da_afmas.py中如何初始化这个判别器：

  grep -n "MultiLevelDomainDiscriminator" 25_da_afmas.py

  然后查看具体的初始化代码：

  sed -n '200,250p' 25_da_afmas.py | grep -A 10 -B 5 "MultiLevel"
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ grep -n "MultiLevelDomainDiscriminator" 25_da_afmas.py
37:MultiLevelDomainDiscriminator = domain_discriminator.MultiLevelDomainDiscriminator
201:            self.multi_level_discriminator = MultiLevelDomainDiscriminator(
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sed -n '200,250p' 25_da_afmas.py | grep -A 10 -B 5 "MultiLevel"
            # 假设提取3个层级的特征
            self.multi_level_discriminator = MultiLevelDomainDiscriminator(
                feature_dims=(512, 1024, 2048),
                dropout_rate=dropout_rate
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化新增层的权重"""
        for m in [self.feature_projector, self.task_classifier, self.agent_weight_generator]:
            for layer in m:


(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sed -n '120,180p' 23_domain_discriminator.py

    创新点:
    - 不仅在最后的全局特征上做域适应
    - 还在中间层特征上做域适应
    - 多层级对齐，效果更好（参考MDAN, CVPR 2018）
    """

    def __init__(
        self,
        feature_dims: Tuple[int, ...] = (512, 1024, 2048),  # 不同层的特征维度
        hidden_dim: int = 512,
        dropout_rate: float = 0.5
    ):
        """
        参数:
            feature_dims: 多个层级的特征维度
            hidden_dim: 判别器隐藏层维度
            dropout_rate: Dropout比例
        """
        super().__init__()

        # 为每个层级创建一个判别器
        self.discriminators = nn.ModuleList([
            DomainDiscriminator(
                feature_dim=feat_dim,
                hidden_dims=(hidden_dim, hidden_dim // 2),
                dropout_rate=dropout_rate
            )
            for feat_dim in feature_dims
        ])

        self.num_levels = len(feature_dims)

    def forward(
        self,
        multi_level_features: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        参数:
            multi_level_features: 多层级特征 (level1, level2, level3, ...)

        返回:
            multi_level_logits: 多层级域分类logits
        """
        assert len(multi_level_features) == self.num_levels, \
            f"Expected {self.num_levels} feature levels, got {len(multi_level_features)}"

        domain_logits_list = []
        for discriminator, features in zip(self.discriminators, multi_level_features):
            logits = discriminator(features)
            domain_logits_list.append(logits)

        return tuple(domain_logits_list)

    def compute_loss(
        self,
        multi_level_features: Tuple[torch.Tensor, ...],
        domain_labels: torch.Tensor,
        weights: Tuple[float, ...] = None


⏺ 找到了！代码结构是对的（2048 -> 1024 -> 512 -> 256 -> 2）。

  问题在于 MultiLevelDomainDiscriminator 可能在初始化子判别器时没有传递正确的 feature_dim。

  查看MultiLevelDomainDiscriminator的初始化：

  sed -n '120,180p' 23_domain_discriminator.py


⏺ 看到了！feature_dim=2048 是正确的。但需要看discriminator是如何构建的。查看完整的初始化代码：

  # 查看__init__方法的完整内容
  sed -n '22,70p' 23_domain_discriminator.py
  
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sed -n '22,70p' 23_domain_discriminator.py

    1. 区分来自source domain(内镜)还是target domain(病理)
    2. 通过对抗训练，强制特征提取器学习域不变特征

    架构:
    - 输入: 特征向量 (batch_size, feature_dim)
    - 输出: 域分类logits (batch_size, 2)  [0=内镜, 1=病理]
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dims: Tuple[int, ...] = (1024, 512, 256),
        dropout_rate: float = 0.5
    ):
        """
        参数:
            feature_dim: 输入特征维度 (默认2048适配ResNet/EfficientNet)
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout比例（防止域判别器过拟合）
        """
        super().__init__()

        self.feature_dim = feature_dim

        # 构建多层判别器
        layers = []
        in_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            in_dim = hidden_dim

        # 最后的域分类层
        layers.append(nn.Linear(in_dim, 2))  # 2个域: source vs target

        self.discriminator = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
  
  
  
  之前的sed命令没有生效。需要找到正确的位置修复。先查看域判别器的初始化：

  # 查看域判别器的__init__方法
  grep -A 20 "class.*Discriminator" 23_domain_discriminator.py | head -30

  或者直接查看第92行附近的代码：
  sed -n '80,100p' 23_domain_discriminator.py
  
  
  (base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ grep -A 20 "class.*Discriminator" 23_domain_discriminator.py | head -30
class DomainDiscriminator(nn.Module):
    """
    域判别器 - 用于对抗训练

    作用:
    1. 区分来自source domain(内镜)还是target domain(病理)
    2. 通过对抗训练，强制特征提取器学习域不变特征

    架构:
    - 输入: 特征向量 (batch_size, feature_dim)
    - 输出: 域分类logits (batch_size, 2)  [0=内镜, 1=病理]
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dims: Tuple[int, ...] = (1024, 512, 256),
        dropout_rate: float = 0.5
    ):
        """
        参数:
--
class MultiLevelDomainDiscriminator(nn.Module):
    """
    多层级域判别器 - 在不同特征层次上进行域适应

    创新点:
    - 不仅在最后的全局特征上做域适应
    - 还在中间层特征上做域适应
    - 多层级对齐，效果更好（参考MDAN, CVPR 2018）
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$  sed -n '80,100p' 23_domain_discriminator.py

        前向传播

        参数:
            features: 特征向量 (batch_size, feature_dim)

        返回:
            domain_logits: 域分类logits (batch_size, 2)
        """
        # 确保输入是2D
        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        domain_logits = self.discriminator(features)
        return domain_logits

    def get_domain_accuracy(
        self,
        features: torch.Tensor,
        domain_labels: torch.Tensor
    ) -> float:
        """

  
  









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


try4

⏺ 🎉 恭喜！SimCLR训练完成了！

  训练结果：
  - ✅ 总时间：5.68小时
  - ✅ 最佳loss: 5.0649
  - ✅ 模型已保存！

⏺ 你的命令缺少参数！让我帮你启动正确的DA-AFMAS训练：

  在实验室终端运行这个完整命令：

  # 先停止错误的进程
  pkill -f 25_da_afmas.py

  # 检查SimCLR模型位置
  ls -lh simclr_logs_rtx4090/best_model.pth

  # 启动完整的DA-AFMAS训练（带所有参数）
  nohup python3 25_da_afmas.py \
      --pretrained_weights ./simclr_logs_rtx4090/best_model.pth \
      --source_data ./processed_data \
      --target_data /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k \
      --num_epochs 50 \
      --batch_size 32 \
      --learning_rate 0.0001 \
      > ~/da_afmas_training.log 2>&1 &

  echo "DA-AFMAS训练启动，进程ID: $!"

  # 查看训练进度
  tail -f ~/da_afmas_training.log

  如果报错找不到文件，先检查路径：
  # 检查必要的路径
  ls -lh simclr_logs_rtx4090/best_model.pth
  ls -d processed_data
  ls -d /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k


修复：：：：：：
  # 单行版本（一次性复制这整行）
  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data
  /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 >
  ~/da_afmas_training.log 2>&1 &

  echo "训练进程ID: $!"

  # 等待2秒后查看日志
  sleep 2
  tail -n 20 ~/da_afmas_training.log

再修复

# 查看问题代码
  head -40 25_da_afmas.py | grep -A 5 -B 5 "Volumes"

  快速修复方案：

  # 修复硬编码路径（将Mac路径改为当前目录）
  sed -i 's|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/|./|g' 25_da_afmas.py

  # 验证修改
  grep "domain_discriminator" 25_da_afmas.py | head -5

  # 重新启动训练
  pkill -f 25_da_afmas.py
  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data
  /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 >
  ~/da_afmas_training.log 2>&1 &

  echo "进程ID: $!"
  sleep 3
  tail -n 30 ~/da_afmas_training.log

try5:

  第1步：修复路径
  sed -i 's|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/|./|g' 25_da_afmas.py

  第2步：验证修复
  grep "23_domain_discriminator" 25_da_afmas.py

  第3步：停止错误进程
  pkill -f 25_da_afmas.py

  第4步：重新启动（单行）
  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data
  /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 >
  ~/da_afmas_training.log 2>&1 &

  第5步：查看日志
  sleep 3 && tail -n 30 ~/da_afmas_training.log

try6

⏺ 路径还没修复！先确认执行了第1步：

  sed -i 's|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/|./|g' 25_da_afmas.py

  执行后验证：
  grep "domain_discriminator.py" 25_da_afmas.py

  你应该看到 ./23_domain_discriminator.py 而不是 /Volumes/Seagate/...

  如果还是显示旧路径，可能需要多修复几个地方：
  # 修复所有可能的硬编码路径
  sed -i 's|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py|./23_domain_discriminator.py|g' 25_da_afmas.py
  sed -i 's|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/24_gradient_reversal_layer.py|./24_gradient_reversal_layer.py|g' 25_da_afmas.py
  sed -i 's|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/05_agent_models.py|./05_agent_models.py|g' 25_da_afmas.py

  # 验证修复
  grep "\.py\"" 25_da_afmas.py | grep -E "(23_|24_|05_)"

  修复完成后再启动：
  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data
  /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 >
  ~/da_afmas_training.log 2>&1 &


(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sleep 3 && tail -n 30 ~/da_afmas_training.log
[1]+  Exit 1                  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 > ~/da_afmas_training.log 2>&1
nohup: ignoring input
Traceback (most recent call last):
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/25_da_afmas.py", line 34, in <module>
    spec_discriminator.loader.exec_module(domain_discriminator)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 1022, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1159, in get_code
  File "<frozen importlib._bootstrap_external>", line 1217, in get_data
FileNotFoundError: [Errno 2] No such file or directory: '/Volumes/Seagate/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py'
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sed -i 's|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/|./|g' 25_da_afmas.py
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ grep "domain_discriminator.py" 25_da_afmas.py

    "./23_domain_discriminator.py"
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ 
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ grep "domain_discriminator.py" 25_da_afmas.py

    "./23_domain_discriminator.py"
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sed -i 's|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py|./23_domain_discriminator.py|g' 25_da_afmas.py sed -i 's|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/24_gradient_reversal_layer.py|./24_gradient_reversal_layer.py|g' 25_da_afmas.py sed -i 's|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/05_agent_models.py|./05_agent_models.py|g' 25_da_afmas.py

sed: can't read sed: No such file or directory
sed: can't read s|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/24_gradient_reversal_layer.py|./24_gradient_reversal_layer.py|g: No such file or directory
sed: can't read sed: No such file or directory
sed: can't read s|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/05_agent_models.py|./05_agent_models.py|g: No such file or directory
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ grep ".py"" 25_da_afmas.py | grep -E "(23_|24_|05_)"
bash: syntax error near unexpected token `('
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ 

  ##修复6

1. 修复其他路径：
  sed -i 's|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/24_gradient_reversal_layer.py|./24_gradient_reversal_layer.py|g' 25_da_afmas.py

  2. 继续修复：
  sed -i 's|/Volumes/Seagate/AFMAS_GastricCancer_Dataset/05_agent_models.py|./05_agent_models.py|g' 25_da_afmas.py

  3. 验证所有修复：
  grep "Volumes" 25_da_afmas.py

  如果没有输出，说明所有路径都修复了！

  4. 重新启动训练：
  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data
  /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 >
  ~/da_afmas_training.log 2>&1 &

  5. 查看日志：
  sleep 3 && tail -n 30 ~/da_afmas_training.log




(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 > ~/da_afmas_training.log 2>&1 &

[1] 10895
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sleep 3 && tail -n 30 ~/da_afmas_training.log
[1]+  Exit 1                  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 > ~/da_afmas_training.log 2>&1
--------------------------------------------------------------------------------
输入形状: torch.Size([4, 3, 224, 224])
类别标签: tensor([2, 1, 2, 1])
域标签: tensor([0, 1, 1, 1])
Traceback (most recent call last):
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/25_da_afmas.py", line 535, in <module>
    outputs = model(x, domain_label=domain_labels, training=True)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/25_da_afmas.py", line 283, in forward
    features, agent_features, agent_logits = self.extract_features(
                                             ~~~~~~~~~~~~~~~~~~~~~^
        x, return_agent_features=True
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/25_da_afmas.py", line 241, in extract_features
    logit, feat = agent(x, return_features=True)
                  ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/05_agent_models.py", line 111, in forward
    out = torch.relu(features, inplace=True)
TypeError: relu() got an unexpected keyword argument 'inplace'


try7

新错误！PyTorch版本问题，torch.relu() 不支持 inplace 参数。修复：

  # 修复 relu inplace 错误
  sed -i 's/torch.relu(features, inplace=True)/torch.nn.functional.relu(features, inplace=True)/g' 05_agent_models.py

  或者更简单，去掉 inplace：
  sed -i 's/torch.relu(features, inplace=True)/torch.relu(features)/g' 05_agent_models.py

  验证修复：
  grep "relu" 05_agent_models.py | grep -n "inplace"

  重新启动训练：
  pkill -f 25_da_afmas.py
  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data
  /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 >
  ~/da_afmas_training.log 2>&1 &

  查看日志：
  sleep 3 && tail -n 30 ~/da_afmas_training.log

  执行这些命令！ 🔧




  (base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sed -i 's/torch.relu(features, inplace=True)/torch.nn.functional.relu(features, inplace=True)/g' 05_agent_models.py

(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$  grep "relu" 05_agent_models.py | grep -n "inplace"
2:            out = torch.nn.functional.relu(features, inplace=True)
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ pkill -f 25_da_afmas.py nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 > ~/da_afmas_training.log 2>&1 &

[1] 11429
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sleep 3 && tail -n 30 ~/da_afmas_training.log

[1]+  Exit 2                  pkill -f 25_da_afmas.py nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 > ~/da_afmas_training.log 2>&1
 -c, --count               count of matching processes
 -f, --full                use full process name to match
 -g, --pgroup <PGID,...>   match listed process group IDs
 -G, --group <GID,...>     match real group IDs
 -i, --ignore-case         match case insensitively
 -n, --newest              select most recently started
 -o, --oldest              select least recently started
 -O, --older <seconds>     select where older than seconds
 -P, --parent <PPID,...>   match only child processes of the given parent
 -s, --session <SID,...>   match session IDs
     --signal <sig>        signal to send (either number or name)
 -t, --terminal <tty,...>  match by controlling terminal
 -u, --euid <ID,...>       match by effective IDs
 -U, --uid <ID,...>        match by real IDs
 -x, --exact               match exactly with the command name
 -F, --pidfile <file>      read PIDs from file
 -L, --logpidfile          fail if PID file is not locked
 -r, --runstates <state>   match runstates [D,S,Z,...]
 -A, --ignore-ancestors    exclude our ancestors from results
 --cgroup <grp,...>        match by cgroup v2 names
 --ns <PID>                match the processes that belong to the same
                           namespace as <pid>
 --nslist <ns,...>         list which namespaces will be considered for
                           the --ns option.
                           Available namespaces: ipc, mnt, net, pid, user, uts

 -h, --help     display this help and exit
 -V, --version  output version information and exit

For more details see pgrep(1).

修复8:
⏺ 命令又被合并了！分开执行：

  1. 停止进程：
  pkill -f 25_da_afmas.py

  2. 启动训练：
  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data
  /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 >
  ~/da_afmas_training.log 2>&1 &

  3. 查看日志：
  sleep 3 && tail -n 30 ~/da_afmas_training.log

wenti9
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ pkill -f 25_da_afmas.py

(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 > ~/da_afmas_training.log 2>&1 &
[1] 11934
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ sleep 3 && tail -n 30 ~/da_afmas_training.log
[1]+  Exit 1                  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 > ~/da_afmas_training.log 2>&1
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py", line 171, in forward
    logits = discriminator(features)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py", line 92, in forward
    domain_logits = self.discriminator(features)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/container.py", line 244, in forward
    input = module(input)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x2048 and 512x512)
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ 


修复9

  快速修复：检查并修改域判别器的输入维度

  # 查看域判别器的输入维度定义
  grep -n "512" 23_domain_discriminator.py | head -10

  或者直接修复（将512改为2048）：
  # 修复域判别器输入维度
  sed -i 's/nn.Linear(512, 512)/nn.Linear(2048, 512)/g' 23_domain_discriminator.py

  验证修复：
  grep "nn.Linear(2048" 23_domain_discriminator.py

  重新启动：
  pkill -f 25_da_afmas.py
  nohup python3 25_da_afmas.py --pretrained_weights ./simclr_logs_rtx4090/best_model.pth --source_data ./processed_data --target_data
  /home/ubuntu-user/workspace/External_Validation_Datasets/GasHisSDB_labeled_1k --num_epochs 50 --batch_size 32 --learning_rate 0.0001 >
  ~/da_afmas_training.log 2>&1 &
  sleep 3 && tail -n 30 ~/da_afmas_training.log

went10

(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ ps aux | grep 25_da_afmas.py
ubuntu-+   12581  0.0  0.0   9284  2044 pts/1    S+   13:40   0:00 grep --color=auto 25_da_afmas.py
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ cat ~/da_afmas_training.log
nohup: ignoring input
/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
================================================================================
                                 测试 DA-AFMAS 框架                                 
================================================================================

使用设备: cpu

[测试1] 创建 DA-AFMAS 模型
--------------------------------------------------------------------------------
✓ 模型创建成功
  - Agents数量: 7
  - 总特征维度: 11264
  - 使用条件域判别器: True
  - 使用多层级域适应: True

  - 总参数量: 154,507,319
  - 可训练参数: 154,507,319

[测试2] 前向传播测试
--------------------------------------------------------------------------------
输入形状: torch.Size([4, 3, 224, 224])
类别标签: tensor([0, 1, 2, 1])
域标签: tensor([1, 1, 0, 0])
Traceback (most recent call last):
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/25_da_afmas.py", line 535, in <module>
    outputs = model(x, domain_label=domain_labels, training=True)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/25_da_afmas.py", line 333, in forward
    multi_level_logits = self.multi_level_discriminator(multi_level_reversed)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py", line 171, in forward
    logits = discriminator(features)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py", line 92, in forward
    domain_logits = self.discriminator(features)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/container.py", line 244, in forward
    input = module(input)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x2048 and 512x512)
(base) ubuntu-user@WS7-3:~/workspace/AFMAS_GastricCancer_Dataset$ tail -n 50 ~/da_afmas_training.log
  - 使用多层级域适应: True

  - 总参数量: 154,507,319
  - 可训练参数: 154,507,319

[测试2] 前向传播测试
--------------------------------------------------------------------------------
输入形状: torch.Size([4, 3, 224, 224])
类别标签: tensor([0, 1, 2, 1])
域标签: tensor([1, 1, 0, 0])
Traceback (most recent call last):
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/25_da_afmas.py", line 535, in <module>
    outputs = model(x, domain_label=domain_labels, training=True)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/25_da_afmas.py", line 333, in forward
    multi_level_logits = self.multi_level_discriminator(multi_level_reversed)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py", line 171, in forward
    logits = discriminator(features)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/workspace/AFMAS_GastricCancer_Dataset/23_domain_discriminator.py", line 92, in forward
    domain_logits = self.discriminator(features)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/container.py", line 244, in forward
    input = module(input)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu-user/anaconda3/lib/python3.13/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x2048 and 512x512)
