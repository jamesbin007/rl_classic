# PGASM

PGSIM
1.算法对比该怎么做 

对比算法分为六种：ia2c(IA2C), ia2c_fp(FPrint), ma2c_nc(MADDPG), ma2c_cnet(CommNet), ma2c_pnet(PowerNet), ma2c_cu(ConseNet), ma2c_dial(DIAL).

config_ma2c_pnet_DER6.ini 文件为训练的配置文件，文件解析：
	“ma2c_pnet” ：为算法名称
	“DER6” ：文章中的microgrid-6
例如：配置FPrint实验。
1.将config_ma2c_pnet_DER6.ini 文件复制一份，存放在configs目录下，即原文件目录下，并重命名为config_ia2c_fp_DER6.ini。
2.修改文件中agent的配置为 ia2c_fp
3.将main.py文件中

	将上述红框中的配置修改为：
	default_base_dir = './ia2c_fp_der6'
	default_config_dir = 'configs/config_ia2c_fp_DER6.ini'
	4. 然后运行main.py即可
对比算法只能一个一个运行，因此运行的结果必须实时保存起来，在trainer.py文件中

上述红框就是保存结果的文件。具体逐步保存如下图所示：

备注：
“DER6” ：文章中的microgrid-6
上面提到对比算法可能涉及到microgrid-20，因此除了修改文件名中的“DER6”->” DER20”，还需要修改下图中（Grid_envs.py文件中DER_num的值-> 20）

2.环境怎么配置 （2选1）
	如果你有Anaconda就按照Anaconda配置环境，如果你只有python环境就按照python环境配置。
	Anaconda环境配置：
	1、创建一个python虚拟环境：conda create -n powernet python=3.7
	2、激活虚拟环境：conda activate powernet
	3、安装 pytorch (torch>=1.2.0):
		pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
	4、安装所需要的包：pip install -r requirements.txt
	Python环境配置：
	1、安装 pytorch (torch>=1.2.0):
		pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
	2、安装所需要的包：pip install -r requirements.txt
