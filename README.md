基于时序分析的订单流失预测-文档
===
数据
==
	训练用数据包括友空间2017-4-19至2019-6-24埋点采集的行为数据；总数据的维度（操作次数，特征维度） = （240,000,000，15），一次操作记录和特征维度可见附录[1]
	企业的实际激活时间对应表： 特征维度：（instanceID, tenantID, 激活时间）
	应用分类对应表： 特征维度：（openAppID, 类别），暂时应用被分为三类；0：一般应用，1：重要应用，2：核心应用

方法和实现
==
在这一章，主要描述了问题的特点，选取的解决方案和实现思路细节；具体的方法，问题和数据维度转换可以参照代码注释。
方法综述
=
经过调研发现，新购用户的流失率远高于续约用户，且在流失前被挽留成功率更高；更重要的，市场调查结果表明，新购用户续约/流失的因素有别于续约用户。因此，为了减少模型和输入数据的方差，我们决定分两个模型分别预测新购用户的续约概率和续约用户的再续约概率；经过简单的数据分析，我们发现时间域上的特征对续约与否有很大的影响。比如，不仅总使用次数和续约概率成正相关，而且在时间域上使用率的差别对续约概率有很大的影响；如果一个公司使用次数在时间域上呈上升趋势，它续约的概率大于如果其呈下降趋势。因此，我们决定基于传统机器学习二维度数据（batch size，feature space）的基础上，加入时间维度（batch size, time step, feature space），建立时域分析数据；然后可以利用循环神经网络（recurrent neural network）强大的解释及泛化能力，映射用户行为在时域上的特征规律。








实现细节
=
暂时，我们只讨论新购用户的模型建立；在业务上，当一个客户到期4个月方可得知其续约与否；对于购买一年产品的客户，如果要知道其续约与否，客户需要在2018-3-24之前购买（文档撰写与 2019-7-24）。在这次实验里，我们取用了2017-05-01至2017-12-31所有的新购用户的数据用来训练和检验模型。
训练数据都存储在服务器上，由于数据量巨大，需要对数据进行降维整合处理；我们先从原有的15个特征里选取需要的特征形成新的特征空间F^N: (f_1,f_2,…,f_n)，再定义维度变化

（操作次数，特征空间）->（每一时间区间的唯一特征组合数，特征空间）

我们遍历每一次操作记录，在每一个时间区间，我们取每个独立的F^N和他的出现次数actions，然后把actions放入F^N记录下来。core.PreProcessing模块提供了上述方法的实现过程，现在时间区间取为天。这个模块需要额外数据“应用数据对应表”配合对应用种类进行降维处理。
因为周末，法定节假日公司依法休息，这些日期的操作数量会偏离（skew）训练的结果，我们决定剔除所有法定休息日。这个方法的实现依赖于一个公共免费节假日标注接口，http://api.goseek.cn/Tools/holiday?date=。Add_feature.py 可以为数据组添加一列新特征：日期类型。
因为每个公司的激活时间不一样，我们不可能给所有公司用统一的时间线，所以我们定义每个公司的激活时间为它自己时间线的公元第0天；对应的实现在set_year.py, 这个函数需要额外数据“企业的实际激活时间对应表”，把时间特征从公元日月年转换为公司激活后的天数。
	我们再对所有的类别型数据（categorical data）进行二分化处理（one hot encoding）以防止神经网络学习到类别型数据之间的大小关系（numerical relationship）。至此，数据的维度依然为（特征组合数，特征维度）；我们先我们需要定义维度变换：

（特征组合数，特征空间）->（batch size（公司数），时间区间（天数），特征空间）

我们先把instanceID和时间特征从特征空间取出，各自成为一个新的维度

（batch size（公司数），时间区间（天数），-1，特征空间）

-1为自由维度，因为每个公司每一天都会有很多不同的特征组合记录。在每个公司时间组合，我们定义（-1，特征空间）矩阵M，并拿出特征actions 为向量V。我们记录 V^T∙M为每个（公司，天）的信息。因为V^T∙M是一个长度为（原特征空间-1） 的向量，数据最后的维度为（batch size（公司数），时间区间（天数），特征空间）。这个方案的实现在dataloader_1.py中。
	因为我们选取的特征之间差距很大，作为全局正则化（normalize）到[ 0 , 1 ]的替换，我们为每一个特征定义独立的正则化方法，这样可以把所有特征的平均大小拉到一个水平线上。
	最后，我们用Keras快速设计了一个基于LSTM的神经网络模型，具体实现在LSTM.py。

未来工作
==
数据增强方法
=
由于在本次实验中仅取了半年内的新购用户，公司数量有限造成神经网络过拟合问题较为严重。我们可以未来通过数据增强的方法增加训练数据数量，减少分类器方差，提高准确率，这两篇论文  提供了非常好的思路，第二篇的结果尤为出色，场景也十分符合我们的问题。
模型参数调整
=
	因为时间问题，模型参数还没有进行系统调整；由于数据量的限制，主要问题是模型的过拟合问题，参数调整应该着重于dropout layer的选择与调整。由于数据的分布不是很复杂，LSTM的参数数量也应该控制在一个比较小的范围而防止过拟合。
需要解决的问题
=
	把时间特征从公元日月年转换为公司激活之后的天数的方法需要重新设计或者优化。现在的方法是用：原有时间特征上的时间戳-企业激活时间戳，这种方法的弊端是，由于所有休息日都被剔除，时间特征上会出现断层，而且每个企业的断层位置还不同。现在的解决方法是把每个企业的变换完的时间转换为其（去重后的时间集合的index）。解决方法具体代码和实现方法在data.py 第91~123行。
	dataloader_1的写法过于臃肿，速度很慢，需要重写。









附录
==
	 ##训练用行为数据维度
{
  "action": "click",
  "app_id": "190",
  "client": "android",
  "client_ip": "10.78.252.142",
  "device_model": "HUAWEI TAG-TL00",
  "device_name": "\u534e\u4e3a",
  "instance_id": "57281",
  "member_id": "2024795",
  "mtime": "1493481601109",
  "object_id": "0",
  "qz_id": "60774",
  "user_id": "2024705",
  "ver_code": "4.0.2",
  "wtime": "1493481601109",
  "open_appid": "-5"
}
	##文件名和结构
模块或文件名称	用途
PreProcessing	数据初步整合
Add_feature.py	增加日期类型特征
Set_year.py	日期对齐
Dataloader_1.py	数据最终维度变换
Data.py	上面三个py文件的主函数
LSTM.py	数据正则化，模型实现





Built With
===

* [numpy](http://www.numpy.org/) - Package for scientific computing with Python
* [pandas](https://pandas.pydata.org/) - data structures and data analysis tools for Python






Authors
===

* **Sida Shen** - *Initial work* - [ss892714028](https://github.com/ss892714028)



Acknowledgments
===

* Inspiration
* etc
