# -AI-
参赛作品名：

网抑云选手等级鉴定器

作品简介：
使用Paddle框架和PaddleNLP根据输入的感悟语句鉴定你是几级网抑云选手？深夜测试有buff加成！
深夜总是打开网易云的时候，基于网易云的热评数据集与情感分析为大家制作了一个网易云选手鉴定器，可以通过你输入的话语与感悟鉴定你是几级网易云选手，使用方法简单，



使用方式：

**步骤如下：

1.在文件目录中找到test.txt文件，点击进入写两句自己的任意感受与想法（文件中已有一些例子，看了后一定会有些许感悟）

2.书写感悟完毕后点击运行将代码全部运行好，拉取在最下方可以查看检测结果

环境介绍

    PaddlePaddle框架，AI Studio平台已经默认安装最新版2.0。

    PaddleNLP，深度兼容框架2.0，是飞桨框架2.0在NLP领域的最佳实践。

训练集展示：
['0', '1']
['你眼里有春有秋 ,胜过我见过爱过的山川河流', '1']
['你曾是我的他却不再是我的他，谢谢你赠与我空欢喜', '0']
['我们都是，苦尽甘来的人，但愿殊途同归，你能与我讲讲来时的路', '0']
['在我的世界里，你的出现让我明白了什么是陪伴。我还记得你说过的这句话“从前车马很慢，一生只够爱一人。”我想做那个人，和你一起携手走完这一生。我知道以后会经历各种困难，但我想跟你并肩同行，和你有着耳鬢厮磨的爱情', '1']
['春意渐浓，想你、念你、陪你、爱你。', '1']
['我想陪你走过春夏秋冬  陪你感受爱恨情长', '1']
['我想陪着你，从执拗到素淡，从青丝到白发，从一场秋到另一场秋，从不谙世事到步履阑珊，我想陪着你，在有限的生命里', '1']
['你曾是我的他却不再是我的他，谢谢你赠与我空欢喜', '0']
['有时关不上冰箱的门， 脚趾撞到了桌腿， 临出门找不到想要的东西， 突然忍不住掉泪， 你觉得小题大作， 只有我自己知道为什么；', '0']
['人总是贪婪的，就像最开始我只想知道你的名字', '0']

测试效果展示：
Data: ['“我总是慢半拍,追不上你的节奏,跟不上你的步伐.”', '0'] 	 网抑云等级: 6
Data: ['你不要嫌弃我衣上烟味，寂寞时谁也不会皱着眉。', '0'] 	 网抑云等级: 6
Data: ['用最真诚的心，谱最纯粹的曲，唱最动人的情，', '1'] 	 网抑云等级: 6
Data: ['在感情快速消费的时代，所谓的暧昧，换不来真心的情感。', '0'] 	 网抑云等级: 6
Data: ['反正现在的感情都暧昧，付出过的人排队谈体会，感情像牛奶一杯 越甜越让人生畏，弃之可惜 食而无味”', '0'] 	 网抑云等级: 6
Data: ['我猜暧昧的意思是，在阳光温和的日子里，爱未曾来。', '1'] 	 网抑云等级: 6
Data: ['“暧昧是什么？”“所有人都以为你们在一起了，只有你清楚的知道你们的距离。', '0'] 	 网抑云等级: 6
Data: ['开始慢慢懂爱情，却开始害怕听情歌。', '0'] 	 网抑云等级: 6
