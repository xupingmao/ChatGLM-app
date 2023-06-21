<!--
 * @Author: xupingmao
 * @email: 578749341@qq.com
 * @Date: 2023-06-21 20:48:08
 * @LastEditors: xupingmao
 * @LastEditTime: 2023-06-21 21:11:18
 * @FilePath: \ChatGLM-app\README.md
 * @Description: 描述
-->
# ChatGLM-app

这是一个demo玩具，用于实验chatglm的一些功能


# 安装

TODO

# 安装过程中遇到的问题


## Library cudart is not initialized

```
RuntimeError: Library cudart is not initialized
```

这个错误是由于cuda的lib不在环境变量中，需要通过程序添加一下。修复代码如下

```python
torch_dir = os.path.dirname(torch.__file__)
os.environ['PATH'] = os.environ.get("PATH", "") + os.pathsep + os.path.join(torch_dir, "lib")
```