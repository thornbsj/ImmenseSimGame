import warnings
warnings.filterwarnings("ignore")
import re
import pandas as pd
import numpy as np
import math
import pickle
import os
import gradio as gr
import time
import PIL
import base64
import io
import json
from openai import OpenAI
import configparser

def base64_to_PIL(base64_string):
    data = base64.b64decode(base64_string)
    buffer = io.BytesIO(data)
    res = PIL.Image.open(buffer)
    return res
bgms = {'bgm1':44100,
    'bgm2':32000,
    'bgm3':32000,
    'bgm4':44100,
    'bgm6':48000,
    'bgm7':44100,
    'bgm8':44100,
    'bgm9':32000,
    'bgm10':32000,
    'bgm11':44100,
    'bgm12':44100,
    'attack':44100,
    'truth':32000,
    'critical':44100}

df_prompt_dtype={'location': "str",
 'condition': "str",
 'prompt': "str",
 'deep': "int"}

df_stage_end_dtype = {'location': "str",
 'key type': "str",
 'key ID': "str",
 'key status': "str",
 'display text': "str",
 'final_end': "float64"}

df_npc_dtype = {'ID': "str",
 'name': "str",
 'strength': "int",
 'sense': "int",
 'patient': "float64",
 'health': "int",
 'prompt': "str",
 'anger_condition': "str",
 'battle_able': "float64",
 'appear': "str",
 'disappear': "str",
 'disappear_cause': "str",
 'disappear_anger': "str",
 'necromancy': "str",
 'necromancy_dead': "str",
 'necromancy_dead_cause': "str",
 'init_word': "str",
 'presets': "str",
 'persuade_key': "str",
 'persuade_value': "float64",
 'persuation_result': "str",
 'persuation_result_txt': "str",
 'persuated_prompt': "str",
 'CSS_class': "str",
 'profile_img': "str",
 'ancestor': "str"}

df_locations_dtype = {'location': "str",
 'item': "str",
 'item_ID': "str",
 'npc_ID': "str",
 'npc_experience': "float64",
 'init': "float64",
 'description': "str",
 'condition': "str",
 'threshold': "float64",
 'wordview': "str",
 'bgm': "str"}

df_events_dtype = {'ItemID': "str",
 'StatusID': "str",
 'location': "str",
 'item': "str",
 'action_keyword': "str",
 'condition': "str",
 'threshold': "str",
 'display': "str",
 'add_history': "str",
 'necromancy': "str",
 'necromancy_result': "str",
 'display_fail': "str",
 'result': "str",
 'result_fail': "str",
 'is_breakable': "int",
 'break_condition': "str",
 'break_threshold': "float64",
 'break_fail': "str",
 'break_fail_result': "str",
 'break': "str",
 'break_result': "str"}

profiles = None
illustrations = None
css = None

model = None
similar_model = None
key = None
client = None
similar_client = None
df_locations = pd.DataFrame()
df_events = pd.DataFrame()
df_npc = pd.DataFrame()
df_stage_end = pd.DataFrame()
df_prompt = pd.DataFrame()
copy_icon = 'iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAABx0lEQVR4nO3Wz4uNURgH8M/MpaEIEeXHQlFKE9la20iZZGFlFEsLC8m/IDvJWspONI0oC8XKzo/d2FjMtRBTJhY3XvdanPOS17nvPYe7muZbb53OeZ7n+/w6z3lZxUrHRIHsBmwukO/jA36UONTELBbxreDr4QWmUwZzIj6EV3iAOQwynV2Hy1jG0ehMEfbiEtaUKuI8KuxsHrQZO4jDcd3FGUyOIFrEM6G+hCj7GXq/MCvUqFJW1wFuY220czbu724SpCLeiGuYxxX5tZnECdzCHTxtE04Rb8I2oZneZZLWuI8b2DNKcFiN++jE9SPsyyCdwZKQ7pGdn9Opz/E2w9iygiZqI67v+D1MZdhawtZxENcRzuNAhq0jeD8O4jri01hvdKoXsGUcxDXRMWzPsHXT78HxX8Q1jmN/Yr855+/iU2OvEhrurzchJ9UzUXnQOPvuzwgr7IrrvpD2U9GZzyXENdGcdMQX8XCIbk8YuydxAV/+hfgJXifOuy06U3gsjM03KeMp4oGQ2vruXm9xLoVO1F9oE0oRfxQe/qvRSC+TcEJ4JCq8zPWyiWnht6Wn7Fns4lyul8PQwQ4F81fo3q8F8qtYwfgJAxV5E7QkzlYAAAAASUVORK5CYII='
checked = 'iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAABzUlEQVR4nOXWPWtUQRTG8d+KFlGILBpXBN28NAYbwVYDBhFfagvB7WySwjZ+AUtBI+IXsBGNRtHOWts0EUFUtMqq2Ai2sZi5OLvevS/uwhb5l2fOOc/cuXPOHHYajZr+h3ESx3Eo2r7hPTawNWrhc1jCCfzEZ3Rj/BRmcACbeIDXVTcwiDae4h1WMFfgOxd9NmNM+39FF/AJd9GsEdfEnRi7UFf0jHCUnbqBCZ2Yo7L4UXwdUjTjWsx1rIrzOu6NQDRjFc/LnBbxAftHKDwZcy4WOa0LN3MYJnAb5xPbSsydSyvubHoI0X14hm1cTeztmLuVF3QBb9XvZhmTeBVFl/rWGniDi5lhV7I4j48xsC5NrOESrgvdK2VbqOv5zLA7WZwS6q4uB/FEqNcOHg7w60Zf9H4x+cfcwC0s56y18AKnhX86SPSf3Knwd39fnH5mcB83EtsRvMQpXMGjAlEx94+8haLLtRePhX+1LHSiDfzG5RLBjJ7LlVJWThOJ+BZ+6a3VIqYVlBOhyG8WJMjEuzhbUZSSBkK1lrlHwc5zqNQyxZ2t1khcRqVHgvAsfjGGZ5ExDQIZYxl9MvqHvdkC31k1hr2xjbdjG+h3Hn8AkzVnpOXQA88AAAAASUVORK5CYII='
quiet = 'iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAMAAAAM7l6QAAAANlBMVEX///8jHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyD///9itox/AAAAEHRSTlMAMGCPv9+vgEAQUJ9wIO/P7UZakQAAAGBJREFUeJzNkjEOQCEIQ5n8a+9/Wv3RRAhQ4mLsZp8Wgoi8pI9BDFGY4okS3NY7jwGV6rCtGOD89Bv+MpQdYeyUAItu5hgX4aY1OGwNOpZiqFP5l6gQgot12AlEdBVvqgMSHQe7AMOhAAAAAABJRU5ErkJggg=='
unquiet = 'iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAMAAAAM7l6QAAAANlBMVEX///8jHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyAjHyD///9itox/AAAAEHRSTlMAMGCPv9+vgEAQUJ9wIO/P7UZakQAAAGhJREFUeJxjYBhEQACvJBDglQRLY1UDlOJmgEpjKEDViCQvIAA3FWGugAALmo1waT4BFHWodkBNFhBgwyENU0SuNIgiX5oSu9E8hjdYcAUqG7KhuKMESQFUmglrqsCfHBA24Ab4ZekJACUaCxCX8RXOAAAAAElFTkSuQmCC'
hurt_img  = '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'
critical_img  = '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' 
class Game:
    def __init__(self,df_locations,df_events,df_npc,df_stage_end,df_prompt):

        self.min_sp = 5
        self.min_elo = 5
        self.status = {
            "strength":self.min_sp,
            "sense":self.min_sp,
            "eloquent":self.min_elo,
            "health":500
        }
        self.points = 10
        self.key_history = []
        self.history=[]
        self.npcs = df_npc
        self.available_npcs = {}
        self.history_npcs = {}
        self.tools=[
            {
            "type": "function",
            "function": {
              "name": "interaction_att",
              "description": "与物品进行非破坏性的交互时，调用此函数",
              "parameters": {
                "obj":{
                  "type": "string",
                  "description": "玩家交互的对象，一定是中文"
              }
                            }
            }
            },
            {
            "type": "function",
            "function": {
              "name": "attack",
              "description": "当玩家想要攻击某个NPC或者破坏某样东西时，调用此函数，但是请注意撬门这件事不算破坏性行为，不应当调用这一函数",
              "parameters": {
                "obj":{
                  "type": "string",
                  "description": "玩家要攻击或破坏的对象"
                    }
                  }
                }
            }
            ,
            {
            "type": "function",
            "function": {
              "name": "goto",
              "description": "玩家要进入另一个场景时调用此函数，需要注明要去的场景名称",
              "parameters": {
                  "s":{
                  "type": "string",
                  "description": "玩家希望能够进入的场景"
              }
                  }
                }
            },
            {
            "type": "function",
            "function": {
              "name": "necromancy",
              "description": "当玩家想要对某个事物或人物使用通灵术时，调用此函数；人物的愤怒不会影响通灵术的使用",
              "parameters": {
                  "obj":{
                  "type": "string",
                  "description": "使用通灵术的对象"
              }
                  }
                }
            }
                ]

        self.available_objects = []
        self.added_presets = []
        self.available_object_id = {}
        self.object_status = {}
        self.location_conditions = df_locations
        self.available_locations = []
        self.errLog = []
        self.stage_history = []
        self.stage_end = df_stage_end
        self.function_dicts = {"interaction_att":self.interaction_att,"attack":self.attack,"goto":self.goto,"necromancy":self.necromancy}

        self.tool_prompt = self.generate_TOOL_PROMPT()
        self.bakcpack_items = ["衣服"]#物品,最多有9个；但是其中有个固定物品（衣服），否则Prompt可能有问题
        self.current_prompt = ""
        self.current_command = ""
        self.events = df_events
        self.threshold = 0.75 #超过这个阈值，那么互动的对象会自动被“纠正”为available_objects或npc中的值
        self.altert = 1.5 #被偷袭方和偷袭方的感知比大于等于这个值，会偷袭失败
        self.sneak_weaken = 0.7 #被偷袭方的体质乘以这个系数
        self.max_dodge = 0.5 #最大闪避概率
        self.max_critical = 0.2 #最大暴击率
        self.critical_hurt = 0.2 #暴击的额外倍率
        self.recover_value = 50 #进入下一个区域时恢复的体力
        # self.Start()
        self.new_display_history=[]
        self.battle_log = []
        self.df_prompt = df_prompt
        self.batting = False #是否正在战斗
        self.current_bgm=None #现在的bgm
        self.is_changed_bgm = False #是否更改了bgm
        self.easter_egg_texts = {
            "枫桥,曹杨,隆德,江苏,交通,徐家……":False,
            "点击录制按钮即可录制本局比赛……虽然既没有按钮也没有比赛就是了。":False,
            "实验失败，未能觉醒自主意识。<br>屏幕前的那位，游戏剩下的部分也拜托你了。":False,
            "暗月从未嚎叫，但是白日却引人咆哮。":False,
            "那2538件物品最终一定会聚集到一起，届时没有世界能够幸免。":False,
            "狐狸和狗最终都会被蛇咬死，但是最后蛇也活不久了。":False,
            "<img src=\"data:image/jpeg;base64,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\">":False
        }
        self.is_Snake_dead = False
        self.sacrifice = False
        self.ending_text = False
        self.end_of_game = False

    def add_chat_history(self,npc_id,content):
        # 广播将当前谈话内容增加给所有其他NPC
        for k,v in self.available_npcs.items():
            if k != npc_id and v.status["status_ID"]!=0:
                v.history.append({"role":"system","content":content})

    def display(self, text,elem_class="",style="", mode="typewriter",is_battle=0,battle_info=0):
        if pd.isna(text):
            return
        print(text)

    def attack(self,obj):
        """
        如果想要破坏或者攻击某个事物
        """
        if self.location == "大漠地下":
            self.display("这里强烈的安心感竟盖过了你原本内心的暴戾之气，你不再想要做出任何破坏性的行为。")
            return

        if self.location == "恐怖分子基地":
            self.display("在恐怖分子的基地里进行攻击操作显然不是什么明智之举；周围的恐怖分子立刻举枪向你射击。")
            self.die()
            return
        if self.location == "另一个世界":
            self.display("你不知道眼前这个人拿的像火铳一样的东西是什么，但是你觉得你还有机会能够进行反击。随着眼前人轻轻的一个扣击动作，你感到了剧烈的疼痛。")
            self.die()
            return
            
        # 首先判断是物品还是npc
        tgt = self.similarest_obj(obj=obj,type="object")
        if not tgt:
            # 是npc
            npc = self.similarest_obj(obj=obj,type="NPC")
            if not npc:
                #  既不是物品也不是npc
                self.display(f"虽然心中对{obj}积攒了许多怒火，但是眼下不是发泄它们的时候")
                return
            if self.location == "处决场":
                self.display("你扣下扳机，发现枪里的竟是哑弹。沙利叶微笑着掏出了他的枪，火焰绽成一朵诡艳的花。")
                self.die()
                return
            if npc.id == "7":
                self.display("在这个你毫不熟悉的世界里，最好先和这个“好人”合作一下吧。")
                return
            if npc.id == "99":
                self.display("尽管你已失去躯体，但是你仍然朝着绿衣人挥出了实际上不存在的拳头。然而拳头径直穿过了绿衣人的脸庞。")
                self.display("<div class=\"Walter\">真有意思，你还是没搞清楚啊，我已经和梵脉融为一体，我已经赢了！</div>")
                if self.sacrifice:
                    self.end(flag="sacrifice")
                    return
                else:
                    self.ending_text = True
                    self.display("尽管如此，你依然还有最后一个机会：<br>梵脉，你决心在绿衣人的意志广播前，也融合进梵脉，或许能够在他的基础上，做出一些能够补救的许愿。")
                    self.display("于是，这便是你的最后一个念头，这便是给到梵脉的意志：")
                    return
            # 根据双方感知判定是否袭击
            is_sneak=int(npc.status["status_ID"]==3 or npc.status["sense"]*2<=self.status["sense"])
            self.battle(is_sneak,npc)
        else:
            if self.location == "处决场" and tgt!="鬓狗":
                self.display("你扣下扳机，发现枪里的竟是哑弹。沙利叶微笑着掏出了他的枪，火焰绽成一朵诡艳的花。")
                self.die()
            if tgt == "23":
                self.display("在这个你毫不熟悉的世界里，最好先和这个“好人”合作一下吧。")
                return
            obj = self.available_object_id[tgt]
            
            obj = self.events[(self.events["ItemID"] == obj) & (self.events["StatusID"] == self.object_status[obj])].reset_index().loc[0,:]
            if obj["StatusID"] == '-1':
                res = obj["display"]
                self.display(res)
                # self.add_history(self.current_command,res)
                return
            if obj["is_breakable"] == 1:
                # 不足以破坏的情况
                if not pd.isna(obj["break_condition"]):
                    if self.status[obj["break_condition"]]<obj["break_threshold"]:
                        res = f"【失败：力量<{int(obj['break_threshold'])}】"+obj["break_fail"]
                        self.display(res)
                        self.history.append({"role":"system","content":f"玩家尝试破坏{tgt}，但是失败了"})
                        self.parse_action_cause(obj["break_fail_result"])
                        return
                res = obj["break"]
                if not pd.isna(obj['break_threshold']):
                    res = f"【成功：力量>={int(obj['break_threshold'])}】"+res
                self.display(res)
                self.history.append({"role":"system","content":f"玩家破坏了{tgt}"})
                self.parse_action_cause(obj["break_result"])
                if "goto" not in obj["break_result"] and obj['ItemID'] not in {"46","47","64","65"}: # 这2个item破坏后npc会有自动战斗，战斗结束会goto函数
                    self.change_status(f"{obj['ItemID']},{-1}")
                    # self.add_history(self.current_command,res)
                return
                # self.add_history(self.current_command,res)
            else:
                # 不可破坏
                res = f"尽管你看{obj['item']}十分不爽，但是很明显，拿{obj['item']}泄愤是不理智的。"
                self.display(res)
            # self.add_history(self.current_command,res)
            
    def necromancy(self,obj):
        """
        如果想要对某个事物进行通灵术
        """
        if self.location in ["深层意识","恐怖分子基地","处决场","记忆圣所"]:
            self.display("现在你正在窥探鬣狗的记忆，然而鬣狗并不会通灵术……")
            return
        # 首先判断是物品还是npc
        tgt = self.similarest_obj(obj=obj,type="object_necromancy") #特指一下不能对自己通灵
        if not tgt:
            # 是npc
            npc = self.similarest_obj(obj=obj,type="NPC")
            if not npc:
                #  既不是物品也不是npc
                self.display(f"通灵术必须要在对象的面前发动；然而{obj}和你似乎还有一段距离……")
                return
            # 对npc的通灵术
            npc.necromancy_to()
            self.new_display_history.extend(npc.display_history)
            if npc.status["status_ID"]==0 or npc.id == "7":
                self.parse_action_cause(npc.necromancy_dead_cause)
        else:
            if tgt == "自我":
                if self.location == "地心国" and '66' in self.object_status.keys() and self.object_status['66']=='2':
                    # 还需更新
                    self.display("你感到一阵头晕目眩，然后听到了无数嘈杂的声音：")
                    if '1' in self.history_npcs.keys() and self.history_npcs['1'].status["status_ID"]==0:
                        self.display("我们无冤无仇，但是我死了你也不肯放过我，现在你也要来陪我了！<br>",style=self.history_npcs['1'].CSS_class)
                    if '2' in self.history_npcs.keys() and self.history_npcs['2'].status["status_ID"]==0:
                        self.display("你这狗娘养的！我做鬼也不会放过你和那个绿衣服的！<br>",style=self.history_npcs['2'].CSS_class)
                    if '3' in self.history_npcs.keys() and self.history_npcs['3'].status["status_ID"]==0:
                        self.display("你和那个穿绿衣服的毁了我的客栈，现在报应要来了！<br>",style=self.history_npcs['3'].CSS_class)
                    if '4' in self.history_npcs.keys() and self.history_npcs['4'].status["status_ID"]==0:
                        self.display("你和那个穿绿衣服的毁了我的客栈，现在报应要来了！<br>",style=self.history_npcs['4'].CSS_class)
                    if '12' in self.history_npcs.keys() and self.history_npcs['12'].status["status_ID"]==0:
                        self.display("我画了许多你最后遭受报应的场景，要看看吗？<br>",style=self.history_npcs['12'].CSS_class)
                    if '27' in self.history_npcs.keys() and self.history_npcs['27'].status["status_ID"]==0:
                        self.display("不必紧张，大限已至。<br>",style="BOSS")
                    self.display("""<span class=\'self_1\'>我们牺牲了，用以换取你的崛起</span><br>\
                                <span class=\'self_2\'>不要像那个穿着绿衣服的人一样！</span><br>\
                                <span class=\'self_3\'>妈妈……妈妈在哪儿？</span><br>\
                                <span class=\'self_4\'>你就只是个提线木偶，有个人正坐在屏幕后操控你呢！</span><br>
                                <span class=\'self_5\'>错误，系统即将超载，启动安全协议，正在清理异常进程……</span><br>""")
                    self.goto("寒铁狱城",is_induce=True)
                else:
                    self.display("你尝试对自我通灵，但是只觉得一阵头晕目眩。")
                return
            obj = self.available_object_id[tgt]
            if tgt == "熔炉":
                self.sacrifice=True
            if tgt == "星空": #此处没有处理好，想要根据玩家的三维来判断通灵术的结果，但是由于已经测试到最后了，而且只有这一个案例，故而在此打补丁做特殊处理
                if self.status["sense"]>=9:
                    obj = self.events[(self.events["ItemID"] == obj) & (self.events["StatusID"] == self.object_status[obj]) & (self.events["condition"] == "sense")].reset_index().loc[0,:]
                elif self.status["eloquent"]>=8:
                    obj = self.events[(self.events["ItemID"] == obj) & (self.events["StatusID"] == self.object_status[obj]) & (self.events["condition"] == "eloquent")].reset_index().loc[0,:]
                else:
                    obj = self.events[(self.events["ItemID"] == obj) & (self.events["StatusID"] == self.object_status[obj])].reset_index().loc[0,:]
            
            else:
                obj = self.events[(self.events["ItemID"] == obj) & (self.events["StatusID"] == self.object_status[obj])].reset_index().loc[0,:]
            if obj["StatusID"] == '-1':
                res = obj["display"]
                self.display(res)
                # self.add_history(self.current_command,res)
                return
            if not pd.isna(obj["necromancy"]):
                res = obj["necromancy"]
                self.display(res)
                self.history.append({"role":"system","content":f"玩家成功对{tgt}进行了通灵术"})
                #self.add_history(self.current_command,res)
                self.parse_action_cause(obj["necromancy_result"])
                if '54' in self.object_status.keys() and self.object_status['54']=='2' and '55' in self.object_status.keys() and self.object_status['55']=='2':
                    self.goto("观星台",is_induce=True)
                if '56' in self.object_status.keys() and self.object_status['56']=='2' and '57' in self.object_status.keys() and self.object_status['57']=='2':
                    self.goto("观星台x",is_induce=True)
                return
            else:
                # 不可使用通灵术
                res = f"通灵术不是什么万金油技能，对一个没有灵魂的东西使用时毫无意义的。"
                self.display(res)
            # self.add_history(self.current_command,res)
        

    def generate_TOOL_PROMPT(self):
        TOOL_PROMPT = """
        ## 工具
        你有以下的工具可以使用：
        
        ### 与物品交互

        interaction_att: 与物品进行非破坏性的交互，如观察/触摸/打开/点击/使用等行为时，调用此函数。 Parameters:{"type": "object", "properties": {"obj":{ "type": "string", "description": "交互的对象，一定是中文" }}}
        
        ### 攻击性行为
        
        attack: 当玩家想要攻击某个人物或者破坏某样东西时，调用此函数，哪怕人物被说服了或者人物是玩家的队友 Parameters:{"type": "object", "properties":{"obj":{ "type": "string", "description": "玩家要攻击或破坏的对象" }}}
        
        ### 去别的地方
        
        goto: 玩家要进入或前往另一个场景时调用此函数，需要注明要去的场景名称 Parameters:{"type": "object", "properties":{s":{ "type": "string", "description": "玩家希望能够进入的场景" }}}

        ### 使用通灵术

        necromancy: 当玩家想要对某个事物或人物使用通灵术时，调用此函数；人物的愤怒不会影响通灵术的使用，只有当玩家在给出的指令中指明了要用"通灵术"才调用此函数 Parameters:{"type": "object", "properties": {"obj":{ "type": "string", "description": "使用通灵术的对象" }}}
        
        ## 注意请按照interaction_att>attack>necromancy>goto的顺序作为调用函数的优先级，越是靠前的函数，越优先调用。
        ## 注意“撬开门”、“拆开机器巡逻狗”、“用身体撞击壁画”以及“将红磷点燃扔到巡逻机器狗旁边”这四件事不能算作破坏性的行为，应该调用interaction_att函数而非attack函数
        ## 特别的，如果玩家表达了“使用西洋火镰点燃西域迷香”这一行为或“假装皇帝的声音命令僵尸”这一行为或者“使用通灵术将自身意志广播，告诉周围灵魂星星可以通往“极乐世界””这一行为，应该调用interaction_att函数
        ## 特别的，如果玩家表达了“将线圈绑到高能电池上”这一行为，应该调用interaction_att函数，传入的obj参数为"线圈"
        ## 通灵术没有其他的叫法，催眠不是通灵术，只有当玩家明确要求“通灵术”时，才能调用necromancy函数
        ## 特别的，如果要玩家说要去以下地点："""+str(self.available_objects)+"""，那么不应该调用goto函数而应该调用interaction_att函数，将地点作为参数
        ## 特别的，如果玩家表达了开枪”这一行为，应该调用attack函数
        ## 你并不一定非要调用函数，如果调用函数的话，就不要返回任何文本内容
        """
        return "\n".join([i.strip() for i in TOOL_PROMPT.splitlines()])
    
    def generate_PROMPT(self,wordview):
        ORIGNINAL_PROMPT = f"""你是一个基于文本的冒险游戏系统，用户就是玩家。用户会输入他们会进行的操作，你的目标是要将游戏中的反馈回给用户。
        {wordview}
        注意你需要使用第二人称进行回复，将“你”作为输出的主语。
        不要输出除了游戏内容以外的任何信息，不要写解释也不要输出命令，除非用户让你这么做。
        如果用户输入的命令没有触发任何函数，那么回复需要遵循以下规则：
        1、不能影响以下物品：{self.available_objects}，如果后续新增了物品，也应该在这一范畴中。
        2、用户不能获得任何新的物品
        3、用户不可以用到{self.bakcpack_items}以外的物品
        4、用户所在场景没有{self.available_objects}以外的物品，如果后续新增了物品，也应该在这一范畴中。
        5、如果有涉及到以下物品时，需要调用函数{self.available_objects}，如果后续新增了物品，也应该在这一范畴中。
        6、用户可以对任何事物或人物进行攻击，并且为此会调用后续说明的攻击函数，无需参考任何对话历史的情况，哪怕人物被玩家说服了或者人物是玩家的队友。

        同时需要注意，后续玩家做出的操作导致的结果以及场景变化都会在对话中使用system角色来输入
        """
        return "\n".join([i.strip() for i in ORIGNINAL_PROMPT.splitlines()])
    
    def add_history(self,user,assistant):
        if not user:
            self.history.append({"role":"assistant","content": assistant})
        elif not assistant:
            self.history.append({"role":"user","content": user})
        else:
            self.history.extend([{"role":"user","content": user},
                {"role":"assistant","content": assistant}])
        
    def look_around(self):
        """观察周围
        玩家不会调用这个函数，这个函数是每到一个新的地点展示地点有些什么；并初始化history
        所有的available_objects必须不能重名
        Args:
            game: 当前游戏状态

        Returns:
            检查的结果
        """

        
        df_sub = self.location_conditions[self.location_conditions["location"] == self.location].reset_index()
        res = ""
        new_npcs=[]
        for i in df_sub.index:
            if df_sub.loc[i,"init"]==0:
                continue
            if not pd.isna(df_sub.loc[i,"condition"]):
                if df_sub.loc[i,"threshold"]>self.status[df_sub.loc[i,"condition"]]:
                    continue
                else:
                    res += f"【通过判定：感知>={int(df_sub.loc[i,'threshold'])}】"
            if not pd.isna(df_sub.loc[i,"description"]):
                res += df_sub.loc[i,"description"]+"\n"
            
            if df_sub.loc[i,"item_ID"]!="0" and df_sub.loc[i,"item"] not in self.available_objects and not pd.isna(df_sub.loc[i,"item_ID"]):
                self.available_objects.append(df_sub.loc[i,"item"])
                self.object_status[df_sub.loc[i,"item_ID"]] = '1'
                self.available_object_id[df_sub.loc[i,"item"]] = df_sub.loc[i,"item_ID"]
            if not pd.isna(df_sub.loc[i,"npc_ID"]):
                experience ="" if pd.isna(df_sub.loc[i,"npc_experience"]) else df_sub.loc[i,"npc_experience"]
                new_npcs.append((df_sub.loc[i,"npc_ID"],experience))
        self.tool_prompt = self.generate_TOOL_PROMPT()
        self.current_prompt = self.generate_PROMPT(df_sub.loc[df_sub.index[0],"wordview"])
        self.current_bgm = df_sub.loc[df_sub.index[0],"bgm"]
        self.is_changed_bgm = True
        self.history = [
                {"role": "system", "content": self.current_prompt+self.tool_prompt}
            ]
        if self.location[0:4] == "寒铁狱城":
            self.available_locations=["梵脉"]
        self.display(res[0:len(res)-1],style="text-align:center;",mode="LineByLine")
        for npc_id,exp in new_npcs:
            self.add_npc(npc_id,exp)
        # 自动战斗事件
        if self.location == "九渊地宫外侧":
            self.add_npc('13')
            self.battle(condition=2,enemy=self.available_npcs['13'])
        if self.location == "九渊地宫外侧x":
            self.add_npc('14')
            self.battle(condition=2,enemy=self.available_npcs['14'])
        if self.location[0:3] == "观星台":
            self.add_npc('27')
            self.battle(condition=2,enemy=self.available_npcs['27'])

    def similarest_obj(self,obj,type="object"):
        def get_embeddings(sentences):
            print(sentences)
            completion = similar_client.embeddings.create(
                model=similar_model,
                input=sentences,
                dimensions=1024,
                encoding_format="float"
            )
            return [i["embedding"] for i in json.loads(completion.model_dump_json())['data']]
        def cosine_similarity(vec1, vec2):
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2)
        
        if type in {"object","object_necromancy"}:
            obj_list = self.available_objects.copy()
            if not obj_list:
                return None #没有可以互动的物品
            if type == "object_necromancy":
                obj_list.append("自我")
                encoded_input = get_embeddings([obj]+obj_list)
            else:
                encoded_input = get_embeddings([obj]+obj_list)
        elif type == "escape_ending":
            obj_list = [obj,"逃离此地"]
            encoded_input = get_embeddings([obj,"逃离此地"])
        elif type == "location":
            obj_list = self.available_locations
            if not obj_list:
                return None #没有可以互动的物品
            encoded_input = get_embeddings([obj]+obj_list)
        else:# NPC
            names = {'牙戌':'狱卒',
                 '妖怪':'狱卒',
                 '殷晦':'客栈老板',
                 '墨聆':'画匠'}
            if self.location in ["深层意识","恐怖分子基地","处决场","记忆圣所"]:
                names["队友"] = "鬓狗"
            elif self.location in ["壁画窟","九渊地宫外侧","九渊地宫","主墓室","主墓室"]:
                names["队友"] = "鬣狗"
            if obj in names.keys():
                obj = names[obj]
            obj_list = [j for i,j in self.available_npcs.items()]
            if not obj_list:
                return None #没有可以互动的NPC
            encoded_input = get_embeddings([obj]+[i.name for i in obj_list])
        similar_res = [cosine_similarity(encoded_input[0],j) for j in encoded_input[1:]]
        print({i:j for i,j in zip(obj_list,similar_res)})
        if max(similar_res)<self.threshold:
            return None
        else:
            return obj_list[np.argmax(similar_res)]

    def similarest_action(self,obj):
        #按action_keyword->不带action的顺序进行判定
        #每种情况
        # action_keyword
        def get_embeddings(sentences):
            completion = similar_client.embeddings.create(
                model=similar_model,
                input=sentences,
                dimensions=1024,
                encoding_format="float"
            )
            return [i["embedding"] for i in json.loads(completion.model_dump_json())['data']]
        def cosine_similarity(vec1, vec2):
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2)
        sub_df_events = self.events[(self.events["ItemID"].isin(self.available_object_id.values()))]
        idxs = []
        for i in sub_df_events.index:
            if self.object_status[sub_df_events.loc[i,"ItemID"]] == sub_df_events.loc[i,"StatusID"]:
                idxs.append(i)
        sub_df_events = sub_df_events.loc[idxs,:]
        # 带特定动作的
        obj_list = [i for i in sub_df_events[(~pd.isna(sub_df_events["action_keyword"]))]["action_keyword"]]
        if len(obj_list)>0:
            encoded_input = get_embeddings([self.current_command]+obj_list)
            similar_res = [cosine_similarity(encoded_input[0],j) for j in encoded_input[1:]]
            print(similar_res,[self.current_command]+obj_list)
            if max(similar_res)>=self.threshold:
                return sub_df_events[sub_df_events["action_keyword"]==obj_list[np.argmax(similar_res)]]
        #condition
        obj_res = self.similarest_obj(obj)
        if obj_res is not None:
            sub_df = sub_df_events[(sub_df_events["item"]==obj_res) & pd.isna(self.events["action_keyword"])]
            return sub_df
        return None
    
    def easter_egg(self):
        if sum(self.easter_egg_texts.values()) == len(self.easter_egg_texts):
            choices = list(self.easter_egg_texts.keys())
        else:
            choices = [i for i in list(self.easter_egg_texts.keys()) if not self.easter_egg_texts[i]]
        choice = np.random.choice(choices)
        self.easter_egg_texts[choice]=True
        return choice


    def interaction_att(self,obj:str):
        """与环境交互会触发的自动事件
        按action_keyword->condition->不带condition的顺序进行判定
        Args:
            action: 对交互对象做的动作
            obj: 要交互的对象
        Returns:
            交互结果
        """

        if self.location in ("大漠地下","梵脉","实验室"):
            self.display("眼下只能和面前的这个人交流。")
            return

        if self.location in ("另一个世界","处决场"):
            self.display("在被人威胁时做这件事可不明智。")
            return
        obj = self.similarest_action(obj)
        if obj is None:
            # 就正常的不调用这个函数的结果
            history = self.history[:]
            if len(history) == 0:
                history.append({"role":"system","content":self.current_prompt})
            else:
                history[0] = {"role":"system","content":self.current_prompt}

            completion = client.chat.completions.create(
                model=model,
                messages=history+[{"role": "user",  "content": self.current_command}]
            )
            res = completion.choices[0].message.content

            self.display(res)
            # self.add_history(None,res)
            return

        # 将obj df根据condition多少->threshold要求->最后空白的顺序进行排序
        obj["sort_id"] = [0 if pd.isna(obj.loc[i,"condition"]) else len(obj.loc[i,"condition"].split(","))*100+sum([int(j) for j in obj.loc[i,"threshold"].split(",")]) for i in obj.index]
        obj = obj.sort_values("sort_id",ascending=False).reset_index()
        for i in obj.index:
            flag = True
            event = obj.iloc[i,:]
            if not pd.isna(event["condition"]):
                condition = [j for j in event["condition"].split(",")]
                threshold = [int(j) for j in event["threshold"].split(",")]
                for j,c in enumerate(condition):
                    if self.status[c]<threshold[j] and not pd.isna(event["display_fail"]) and i == obj.index[-1]:
                        self.display(event["display_fail"])
                        self.parse_action_cause(event["result_fail"])
                        return
                    elif self.status[c]<threshold[j]:
                        flag = False
                        break
            if flag:
                break
        # 石碑的特殊函数
        final_display = "" if pd.isna(event["display"]) else event["display"]
        if "<placeholder_for_server>" in final_display:
            final_display = final_display.replace("<placeholder_for_server>",self.easter_egg())
        self.display(final_display) #不能加入history，否则下次就不调用函数了
        self.parse_action_cause(event["result"])
        # event = self.events[(self.events["ItemID"] == obj) & (self.events["StatusID"] == self.object_status[obj])].reset_index().loc[0,:]
        if not pd.isna(event["add_history"]):
            self.history.append({"role":"system","content": event["add_history"]})
        
    def inherit_after_snake_dies(self):
        inherit_chains = {
            '43':'74',
            '45':'76',
            '44':'75',
            '60':'62',
            '59':'63',
            '58':'61',
            '72':'73',
            '68':'70',
            '67':'69',
            '46':'47',
            '54':'56',
            '55':'57',
            '49':'52',
            '50':'53',
            '48':'51'
        }
        # 更新id
        for k,v in self.available_object_id.items():
            if v in inherit_chains.keys():
                self.available_object_id[k] = inherit_chains[v]
                print(f"{v}更新为：{inherit_chains[v]}")
        # 更新状态
        key_delete = []
        for k,v in self.object_status.items():
            if k in inherit_chains.keys():
                key_delete.append((k,v))

        for k,v in key_delete:
            self.object_status[inherit_chains[k]] = v
            del self.object_status[k]
        # 更新当前location
        self.location = self.location+"x"

    def parse_action_cause(self,event_result):
        if pd.isna(event_result):
            return
        change_mv_dict = {
            "add_npc":self.add_npc, #增加npc
            "change_status":self.change_status, #更改物品状态
            "battle":self.battle, #诱发战斗 这个函数需要先解析，再进入
            "goto":self.goto, #进入其他场景 应该限制“不能回到原来的地方”
            "get_item":self.get_item, #获得道具
            "add_location":self.add_location, #增加可以去的地方
            "self_change":self.self_change, #改变自身状态
            "npc_status":self.change_npc_status, #改变npc状态（如果存在的话）
            "add_available_item":self.add_available_item, #增加可互动场景
            "add_preset":self.add_preset, #增加提示
            "add_npc_presets":self.add_npc_presets, #增加说服点
            "add_npc_induce":self.add_npc_induce, #增加诱发
            "display":self.display, #显示
            "rmv_npc":self.rmv_npc #清除npc
        }
        actions  = event_result.split(";")
        for action in actions:
            if len(action.split(":"))>0 and action.split(":")[0]=="add_npc_induce":
                self.add_npc_induce(":".join(action.split(":")[1:]))
                continue
            if len(action.split(":")) != 2:
                break
            function_name,parameter_string = action.split(":")
            if function_name == "battle":
                condition,enemy = parameter_string.split(",")
                if enemy not in self.available_npcs.keys():
                    self.add_npc(enemy)
                enemy = self.available_npcs[enemy]
                self.battle(int(condition),enemy)
            elif function_name == "goto":
                self.goto(parameter_string,is_induce=True)
            else:
                change_mv_dict[function_name](parameter_string)

    def add_preset(self,s):
        if s not in self.added_presets:
            self.added_presets.append(s)
    
    def add_npc_presets(self,s):
        npc_id,preset = s.split(",")
        for npc in self.available_npcs.keys():
            if npc == npc_id:
                self.available_npcs[npc].add_preset(preset)

    def add_npc_induce(self,s):
        npc_id,preset,system_prompt,parse_rst = s.split("###")
        print(npc_id)
        for npc in self.available_npcs.keys():
            if npc == npc_id:
                print(npc)
                self.available_npcs[npc].induce[preset]=[system_prompt,parse_rst.replace("***",";"),0]

    def change_npc_status(self,s):
        # 判断npc是否存在 判断其是否已经死亡/离开 显示诱发的信息并执行诱发条件
        npc_id,npc_status = s.split(",")
        npc_status = int(npc_status)
        if npc_id not in self.available_npcs.keys() or self.available_npcs[npc_id].status["status_ID"] == 0:
            return
        self.available_npcs[npc_id].status["status_ID"]=npc_status
        if npc_status == 0: #导致死亡
            self.display(self.available_npcs[npc_id].disappear)
            self.history.append({"role":"system","content":f"玩家的行为导致了{self.available_npcs[npc_id].name}的死亡"})
        elif npc_status == 3: #被说服
            self.display(self.available_npcs[npc_id].persuation_result_txt)
            self.history.append({"role":"system","content":f"玩家的行为成功地说服了{self.available_npcs[npc_id].name}，{self.available_npcs[npc_id].name}告诉玩家{self.available_npcs[npc_id].persuation_result_txt}"})
    
    def add_available_item(self,s):
        df_sub = self.location_conditions[(self.location_conditions["location"] == self.location) & (self.location_conditions["item_ID"] == s)].reset_index()
        res = ""
        if df_sub.loc[0,"item"] not in self.available_objects:
            self.available_objects.append(df_sub.loc[0,"item"])
            self.object_status[df_sub.loc[0,"item_ID"]] = '1'
            self.available_object_id[df_sub.loc[0,"item"]] = df_sub.loc[0,"item_ID"]
            res = df_sub.loc[0,"description"]

            self.current_prompt = self.generate_PROMPT(df_sub.loc[df_sub.index[0],"wordview"])
            self.history[0] = {"role": "system", "content": self.current_prompt+self.tool_prompt}
            self.display(res,style="text-align:center;",mode="typewriter")

    def add_location(self,s):
        self.available_locations.append(s)
        self.history.append({"role":"system","content":f"玩家现在可以前往{s}了"})
        self.display(f"新增可到达地区：{s}")
    
    def self_change(self,s):
        k,v = s.split(",")
        if k == "health":
            # 特殊处理，星空直接半血
            self.status["health"] = int(self.status["health"]/2)
        else:
            self.status[k] += int(v)

    def goto(self,s:str,is_induce=False):
        # 这个也应该作为基础工具
        if not is_induce:
            s = self.similarest_obj(obj=s,type="location")
        if s not in self.available_locations and not is_induce:
            res = f"{s}在你的脑海中一闪而过，但是你现在无法到达那里"
            self.display(res)
        else:
            if self.location == "南方客栈":
                s = "大漠地下"
            if self.location[0:3] == "壁画窟":
                s = "九渊地宫外侧"
            if self.is_Snake_dead and s in ["壁画窟","九渊地宫外侧","九渊地宫","主墓室","观星台","寒铁狱城"]:
                s += "x"
            self.display_stage_end()
            self.location = s
            # 状态初始化
            self.history=[]
            
            for k,v in self.available_npcs.items():
                self.history_npcs[k] = v
            self.available_npcs = {}
            self.available_objects = []
            self.added_presets = []
            self.available_object_id = {}
            self.object_status = {}
            self.available_locations = []
            # 回血
            self.status["health"] += self.recover_value
            self.chosen_npc = "system"
            self.look_around()
            
    def display_stage_end(self):
        sub_df = self.stage_end[self.stage_end["location"] == self.location]
        for i in sub_df.index:
            if sub_df.loc[i,"key type"] == "object":
                if sub_df.loc[i,"key ID"] in self.object_status.keys() and self.object_status[sub_df.loc[i,"key ID"]] in sub_df.loc[i,"key status"].split(","):
                    self.display(sub_df.loc[i,"display text"],style="text-align:center;",mode="LineByLine")
                    self.key_history.append((sub_df.loc[i,"key type"],sub_df.loc[i,"key ID"],sub_df.loc[i,"key status"]))
            elif sub_df.loc[i,"key type"] == "character":
                if sub_df.loc[i,"key ID"] in self.available_npcs.keys() and str(self.available_npcs[sub_df.loc[i,"key ID"]].status["status_ID"]) in sub_df.loc[i,"key status"].split(","):
                    self.display(sub_df.loc[i,"display text"],style="text-align:center;",mode="LineByLine")
                    self.key_history.append((sub_df.loc[i,"key type"],sub_df.loc[i,"key ID"],sub_df.loc[i,"key status"]))
            elif sub_df.loc[i,"key type"] == "final":
                self.display(sub_df.loc[i,"display text"],style="text-align:center;",mode="LineByLine")

    def change_status(self,s):
        tgt,status_id = s.split(",")
        self.object_status[tgt] = status_id

    def battle(self,condition,enemy):
        if enemy.id == '10':
            status_save = self.status.copy() #暂时将真实的状态拷贝为鬣狗的
            self.status = {
                "strength":15,
                "sense":15,
                "eloquent":15,
                "health":1700
            }
        if enemy.id == '9':
            self.display("沙利叶不耐烦地掏出了他的枪，火焰绽成一朵诡艳的花。")
            self.die()
            return
        with_snake = False
        if not self.is_Snake_dead and enemy.id in {"13","14","17","18","19","20","27"}:
            with_snake = True
        if enemy.battle_able==0:
            self.display(f"攻击{enemy.name}是没有意义的")
            return
        if enemy.status["status_ID"] == 0:
            self.display(f"攻击{enemy.name}是没有意义的,他已经被你打死了")
            return
        loss = 0
        self.display("begin",is_battle=1) #战斗框开始
        if condition==1: #主动偷袭
            if enemy.status["sense"]/self.status["sense"]>=self.altert:
                self.display(f"{enemy.name}察觉到了你，偷袭失败",battle_info={"target":enemy.id,"result":"missing","remaining_hp":enemy.status["health"]})
            else:
                loss = max(math.ceil(self.status["strength"]-enemy.status["strength"]/2*self.sneak_weaken),1)*10
                enemy.status["health"] -= loss
                self.display(f"偷袭成功<br>你对{enemy.name}造成了{loss}点伤害",battle_info={"target":enemy.id,"result":"critical","remaining_hp":enemy.status["health"]})
        elif condition == 2:
                if self.status["sense"]/enemy.status["sense"]>=self.altert:
                    self.display(f"你察觉到了{enemy.name}的偷袭，抵挡住了",battle_info={"target":"self","result":"missing","remaining_hp":self.status["health"]})
                else:
                    loss = max(math.ceil(enemy.status["strength"]-self.status["strength"]/2*self.sneak_weaken),1)*10
                    self.status["health"] -= loss
                    self.display(f"{enemy.name}对你造成了{loss}点伤害",battle_info={"target":"self","result":"critical","remaining_hp":self.status["health"]})
        while True: #self.status["health"]>0 and enemy.status["health"]>0 避免一击必杀后不调用后续function
            # 根据sense投色子，决定暴击和闪避
            # sense值自己决定闪避率，2方差值决定暴击率

            # 判断暴击与闪避
            self_dodge = np.random.random()
            enemy_dodge = np.random.random()
            self_critic = np.random.random()
            enemy_critic = np.random.random()
            snake_defend = np.random.random()
            snake_attack = np.random.random()
            is_dodged = self_dodge<=min(self.max_dodge,self.status["sense"]/10)
            enemy_is_dodged = enemy_dodge<=min(enemy.status["sense"]/10,self.max_dodge)
            is_critic = self_critic<=min(self.max_critical,(self.status["sense"]-enemy.status["sense"])/20)
            enemy_is_critic = enemy_critic<=min(self.max_critical,(enemy.status["sense"]-self.status["sense"])/20)
            # 我方攻击
            if enemy_is_dodged:
                self.display(f"{enemy.name}躲开了你的攻击",battle_info={"target":enemy.id,"result":"missing","remaining_hp":enemy.status["health"]})
            else:
                if with_snake and snake_attack<=0.5:
                    loss = max(math.ceil(self.status["strength"]-enemy.status["strength"]/2*self.sneak_weaken),1)*10
                    enemy.status["health"] -= loss
                    self.display(f"鬣狗瞅准时机，对{enemy.name}发起了袭击。对其造成了{loss}点伤害",battle_info={"target":enemy.id,"result":"hurt","remaining_hp":enemy.status["health"]})
                if is_critic and enemy.status["health"]>0:
                    self.display(f"你看准了{enemy.name}的一个破绽，打出了暴击！",battle_info={"target":enemy.id,"result":"critical","remaining_hp":enemy.status["health"]})
                    loss = max(math.ceil(self.status["strength"]-enemy.status["strength"]/2*(1+self.critical_hurt)),1)*10
                else:
                    loss = max(math.ceil(self.status["strength"]-enemy.status["strength"]/2),1)*10
                enemy.status["health"] -= loss
                self.display(f"你对{enemy.name}造成了{loss}点伤害",battle_info={"target":enemy.id,"result":"hurt","remaining_hp":enemy.status["health"]})

            if enemy.status["health"]<=0:
                enemy.status["status_ID"] = 0
                enemy.status["health"] = 0
                self.display("end",is_battle=2) #战斗框结束
                self.display(enemy.disappear)
                self.parse_action_cause(enemy.disappear_cause)#解析disappear中的命令
                self.history.append({"role":"system","content": f"玩家与{enemy.name}战斗并获胜，{enemy.name}死亡"})
                self.add_chat_history(enemy.id,f"玩家与{enemy.name}战斗并获胜，{enemy.name}死亡")
                if enemy.name == '“鬣狗”':
                    self.is_Snake_dead = True
                    self.inherit_after_snake_dies()
                if enemy.id == '10':
                    self.status=status_save.copy()
                return
            # 敌方攻击
            if with_snake and snake_defend<=0.5:
                self.display(f"鬣狗帮你挡下了{enemy.name}的这次攻击",battle_info={"target":"self","result":"missing","remaining_hp":self.status["health"]})
            else:
                if is_dodged:
                    self.display(f"你躲开了{enemy.name}的攻击",battle_info={"target":"self","result":"missing","remaining_hp":self.status["health"]})
                else:
                    if enemy_is_critic:
                        self.display(f"{enemy.name}看准了你的一个破绽，打出了暴击！",battle_info={"target":"self","result":"critical","remaining_hp":self.status["health"]})
                        loss = max(math.ceil(enemy.status["strength"]-self.status["strength"]/2*(1+self.critical_hurt)),1)*10
                    else:
                        loss = max(math.ceil(enemy.status["strength"]-self.status["strength"]/2),1)*10
                    if with_snake and loss>=self.status["health"]:
                        self.display(f"鬣狗帮你挡下了{enemy.name}的这次致命攻击",battle_info={"target":"self","result":"missing","remaining_hp":self.status["health"]})
                    elif enemy.id=="10" and loss>=self.status["health"]:
                        self.display(f"鬣狗的意志拒绝了死亡，强行挺下了这致命的攻击",battle_info={"target":"self","result":"missing","remaining_hp":self.status["health"]})
                    else:
                        self.status["health"] -= loss
                        self.display(f"{enemy.name}对你造成了{loss}点伤害",battle_info={"target":"self","result":"hurt","remaining_hp":self.status["health"]})
            if self.status["health"] <= 0:
                self.status["health"] = 0
                self.display("end",is_battle=2) #战斗框结束
                self.die() #失败
                return
        return

    def end(self,flag="normal"):
        prompt_for_normal = "你是一个基于文本的冒险游戏系统，用户就是玩家。用户会输入他们会进行的操作，你的目标是要将游戏中的反馈回给用户。\n\
            现在玩家来到了游戏的最后目标——梵脉；在这之前，游戏中的反派“绿衣人”已经和梵脉融合，以此统一所有意志，让所有人都融合化为同一个高维人，实现集体飞升。\n\
            玩家现在要向梵脉许愿,期望能够在他的基础上，做出一些能够补救的许愿。\n\
            注意对梵脉的设定，是可以广播自身意志，所以玩家的许愿某种程度上是能够实现的。\n\
            现在你应当想办法对玩家给出的许愿命令进行合理想象，将最终的结局给到玩家。除此以外，还需要注意以下原则：\n\
            1、结局应当呈现“许愿的两面性”，如：玩家许愿所有人融合后称为善良的好人，或者许愿所有人都变成善良的人，那么你就应当设定结局为“所有人都突然一心向善，但是每个人对善良的定义有所不同，所以依然有着不少的纷争与痛苦。每个人眼中的‘邪恶’依然存在。”\n\
            2、假设玩家输入的命令和“广播自身意志从而改变世人”的许愿无关，你就表达“你最终的念头是这个，浪费了在梵脉的许愿的机会，绿衣人的融合现在已经开始了”\n\
            3、你应当用一种“悲伤”的语气润色文本，包括上面1、2两点里面的内容，不超过100字。"
        if flag == "escape":
            self.display("没有人会责怪你在此地逃离，毕竟绿衣人已经赢了他所设的游戏。")
        if flag == "normal":
            print("进入普通结局")
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "system",  "content": prompt_for_normal},{"role": "user",  "content": self.current_command}]
            )
            res = completion.choices[0].message.content
        # 公用事件：
        if '2' in self.history_npcs.keys():
            if self.history_npcs['2'].status["status_ID"]==0:
                self.display("<p class=\"goblin-fadeIn\">尽管狱卒已经被你杀死，但是牢狱依然时的牢狱，那个关着你的地牢还静静地躺在大漠中，似乎还吸收着过路旅人的灵魂。</p>",mode="LineByLine")
            else:
                self.display("<p class=\"goblin-fadeIn\">在意识到自己已经永远无法回到族群后，妖怪重振旗鼓，将地牢中似乎是来自另一个世界的机器砸烂并将原本自己看守的地牢用黄沙掩埋。</p><br>\
                             <p class=\"goblin-fadeIn\">自此大漠流传着这样一个传说：在你迷失方向或者被大漠强盗拦住去路时，会有一个断了角的妖股来帮助你。</p>",mode="LineByLine")
        if '4' in self.history_npcs.keys():
            if self.history_npcs['4'].status["status_ID"]==0:
                self.display("<p class=\"manager-fadeIn\">大漠中不多的客栈被焚毁了，前来调查的官兵搜索到了另一个世界的机器以及无数人的尸体。</p><br>\
							<p class=\"manager-fadeIn\">这已经超出了他们的理解，在这桩案件悬而未决多年后，终究和大漠的沙子一样被掩埋、遗忘了。</p>",mode="LineByLine")
            else:
                self.display("<p class=\"manager-fadeIn\">从恐惧与癫狂中回复过来后，客栈老板清点了那些在惨剧中死亡的客人。</p><br>\
								<p class=\"manager-fadeIn\">幸而没有达官贵人，不会有人来向他问责。</p><br>\
                             <p class=\"manager-fadeIn\">意外的是，他没有离开自己的客栈，反而将其重新开张。</p><br><p class=\"manager-fadeIn\">他不再贪财、甚至愿意主动帮助那些迷失在大漠中的路人。</p>",mode="LineByLine")
        if '12' in self.history_npcs.keys():
            if self.history_npcs['12'].status["status_ID"]==0:
                self.display("<p class=\"painter-fadeIn\">画匠死在了他最后的作品旁。有人说他已经疯了，他最后的作品明显是他疯癫之后看到的景象；但是也有人说，他是看到了“世界的真相”无法接受，将它画成画之后自杀的。</p><br>\
                             <p class=\"painter-fadeIn\">随着时间的推移，人们惊恐地发现，他生前所作过地每一副画，内容都正在慢慢变化。</p>",mode="LineByLine")
            else:
                self.display("<p class=\"painter-fadeIn\">画匠将壁画窟内的作品擦去。在发现自己的都料匠朋友再未回来后，他对外宣称自己封笔。</p><br>\
                             <p class=\"painter-fadeIn\">当然，这只是他的一面之词。他朝着九渊地宫的方向旅行，沿途在无人的地方疯狂作画。他知道自己朋友的死不简单，也知道在壁画窟遇到的两个人不平凡。他渴望使用自己的能力发现真相。</p><br>\
                             <p class=\"painter-fadeIn\">随着时间的推移，人们惊讶地发现，他之前所作过地每一副画，内容都正在慢慢变化。</p>",mode="LineByLine")
        if '25' in self.history_npcs.keys():
            if self.history_npcs['25'].status["status_ID"]==0:
                self.display("<p class=\"painful-fadeIn\">苦修者倒在了他实现梦想的前一刻。</p><br>\
                             <p class=\"painful-fadeIn\">或许在某个世界里，和他有着相同志向的另一个苦修者成功了。只是没人知道，“所有世界的痛苦”究竟需要多少这样的苦修者才能稀释成功。</p>",mode="LineByLine")
            else:
                self.display("<p class=\"painful-fadeIn\">没人知道苦修者进入熔炉后是否还活着。</p><br>\
                             <p class=\"painful-fadeIn\">但是在很小的一段时间内，所有世界的人似乎都觉得自己获得了久违的宁静。</p><br>\
                             <p class=\"painful-fadeIn\">或许在某个世界的某个时间点上，会有什么人意识到苦修者的存在，又会有新的人，将人们的痛苦稀释。</p>",mode="LineByLine")
        if '26' in self.history_npcs.keys():
            if self.history_npcs['25'].status["status_ID"]==0:
                self.display("<p class=\"painful-fadeIn\">苦修者倒在了他实现梦想的前一刻。</p><br>\
                             <p class=\"painful-fadeIn\">或许在某个世界里，和他有着相同志向的另一个苦修者成功了。只是没人知道，“所有世界的痛苦”究竟需要多少这样的苦修者才能稀释成功。</p>",mode="LineByLine")
            else:
                self.display("<p class=\"painful-fadeIn\">没人知道苦修者进入熔炉后是否还活着。</p><br>\
                             <p class=\"painful-fadeIn\">但是在很小的一段时间内，所有世界的人似乎都觉得自己获得了久违的宁静。</p><br>\
                             <p class=\"painful-fadeIn\">或许在某个时间点上，会有什么人意识到苦修者的存在，又会有新的人，将人们的痛苦稀释。</p>",mode="LineByLine")
                
        if self.is_Snake_dead:
            self.display("<p class=\"snake-fadeIn\">“鬣狗”从假死中复苏过来，拍了拍身上的灰尘。</p><br>\
                         <p class=\"snake-fadeIn\">任务日志：虽然目标的体能很强，但是通过定位器已经成功定位到梵脉的位置了。</p>",mode="LineByLine")
        else:
            self.display("<p class=\"snake-fadeIn\">将自己扮作“鬣狗”的特工拿出对讲机。</p><br>\
                         <p class=\"snake-fadeIn\">任务完成：找到了梵脉的位置。接下来，就是根据此地定位其他梵脉了。</p>",mode="LineByLine")
        if flag == "normal":
            self.display(res)
        if flag in ("normal","escape"):
            self.display("你忽然感受到成千上万个意识涌入自己的脑中。<br>在你意识到什么事情发生之前，你的记忆已经随着意识的涌入渐渐褪去。你感受到所有的记忆与灵魂正在与你融合。<br>直到你挣扎着撑开眼帘，发现自己身处地牢之中。",mode="LineByLine")
        if flag == "sacrifice":
            self.display("你痛苦的灵魂宛如炸药一般，点燃了梵脉中的星空，灵魂向外部逃逸，只剩你和绿衣人“燃烧”着。",mode="LineByLine")
            self.display("<p class=\"Walter-fadeIn\">为什么？你为什么要这样做？哪怕永远消逝，也要让所有人被困在这个维度，各自分裂，互为敌意？</p>",mode="LineByLine")
            self.display("你因为痛苦无法回答他，但是你已经能够想象到他瞳孔放大、一脸不可置信的标签。<br>这就够了，他能有这种结局，对你而言就够了。",mode="LineByLine")
            self.display("你感到灵魂滚烫，自己的所有记忆都随之消逝，马上就连自己的存在也要被消磨。",mode="LineByLine")
            self.display("然而你立刻又感到一阵阴冷的感觉，甚至还有一丝阴风。<br>你挣扎着撑开眼帘，发现自己身处地牢之中。",mode="LineByLine")

        self.end_of_game = True
        return
    
    def die(self):
        end_string = """随着一阵恍惚，你惊异于自身的无力；体内的热量一点点流失，身体变得冰冷，视线变得模糊。
                     “还不行，我还有事没有完成”
                     尽管你百般拒绝，你体内的力量依然一点点地消逝。最终，伴随着不甘与痛苦，你的眼中的场景完全变黑了。"""
        self.display("\n".join([i.strip() for i in end_string.splitlines()]))
        self.location = ""
        self.min_sp = 5
        self.min_elo = 5
        self.status = {
            "strength":self.min_sp,
            "sense":self.min_sp,
            "eloquent":self.min_elo,
            "health":500
        }
        self.key_history = []
        self.history=[]
        self.available_npcs = {}
        self.history_npcs = {}
        self.available_objects = []
        self.added_presets = []
        self.available_object_id = {}
        self.object_status = {}
        self.available_locations = []
        self.stage_history = []
        
        self.bakcpack_items = ["衣服"]#物品,最多有9个；但是其中有个固定物品（衣服），否则Prompt可能有问题 暂时未用到
        self.current_prompt = ""
        self.current_command = ""
        self.is_Snake_dead = False
        self.sacrifice = False
        self.end_of_game = True
        self.ending_text = False
        # self.Start()

    def add_npc(self,npc,new_experience=""):
        if npc not in self.available_npcs.keys():
            if npc in self.history_npcs.keys():
                self.history.append({"role":"system","content":f"人物回归：{self.history_npcs[npc].name}"})
                new_npc = self.history_npcs[npc]
            else:
                npc_para = self.npcs[self.npcs["ID"]==npc].reset_index().loc[0,:]
                system_info = f"新增人物：{npc_para['name']}"
                names = {
                 '客栈老板':'殷晦',
                 '画匠':'墨聆'}
                if npc_para['name'] == "狱卒":
                    system_info += ",他是一个妖怪，名叫牙戌"
                if npc_para['name'] in names.keys():
                    k = npc_para['name']
                    system_info += f",他名叫{names[k]}"
                self.history.append({"role":"system","content":f"新增人物：{system_info}"})
                npc_para=dict(npc_para)
                if not pd.isna(npc_para["ancestor"]):
                    npc_para["ancestor"] = self.history_npcs[npc_para["ancestor"]]
                new_npc = Character(**dict(npc_para))
            
            if len(new_experience)>0:
                new_npc.history.append({"role":"system","content": new_experience})
            print(f"add_npc:{npc}")
            self.available_npcs[npc] = new_npc
            self.new_display_history.extend(new_npc.display_history)

    def rmv_npc(self,npc):
        if npc in self.available_npcs.keys():
            del self.available_npcs[npc]

    def get_item(self,s:str): #这个暂时不用
        # 不做“物品”表
        self.bakcpack_items.append(s)
        self.display(f"你获得了{s}")
        
    def check_status(self): #这个暂时不用
        """查看当前游戏状态，如果有必要就更新

        Args:
            game: 当前游戏状态

        Returns:
            当前状态，如果已经输了或者赢了就结束
        """
        pass
    
    def merge_system_info(self):
        # 由于OpenAI接口获得多个system+user输入后会直接stop返回空，所以此处将所有的system提示放在一起
        res = self.history[0]["content"] #最初写的prompt
        flag = False
        for d in self.history[1:]:
            if "role" in d.keys() and d["role"] == "system":
                if flag:
                    res += "\n以下是现在游戏的重要信息记录：\n"
                    flag = False
                res += d["content"]+"\n"
        return {"role":"system","content":res[0:-1]} #最后一个换行符不要

    def give_command(self,command):
        if command.strip()=="":
            return
        self.current_command = command
        if self.ending_text == True:
            self.end(flag="normal")
            return
        if self.location[0:4]=="寒铁狱城": #逃离结局
            ending_flag = self.similarest_obj(command,type="escape_ending")
            if ending_flag is not None:
                print("进入逃离结局")
                self.end(flag="escape")
                return
            
        # 由于OpenAI接口获得多个system+user输入后会直接stop返回空，所以此处将所有的system提示放在一起（但是直接用huggingface上Download下来的模型就没有这个限制？）
        completion = client.chat.completions.create(
            model=model,
            messages=[self.merge_system_info()]+[{"role": "user",  "content": command}],
            tools=self.tools
        )
        res = completion.choices[0].message.content
        print(res)
        # 有command的情况
        if completion.choices[0].message.content.strip() == "" and completion.choices[0].message.tool_calls is not None:
            print(f"【调试模式】:{completion.choices[0].message.tool_calls}")
            try:
                fn_name = completion.choices[0].message.tool_calls[0].function.name
                fn_args = completion.choices[0].message.tool_calls[0].function.arguments
                print("执行函数：",fn_name,fn_args)
                self.function_dicts[fn_name](**dict(json.loads(fn_args)))
                return
            except Exception as e:
                self.errLog.append((self.history,e))
                self.display("不知为何，有一股莫名的力量阻止了你的这一行为，你感受到有什么强大的意志阻挠你，最好做些别的事情")
                return
        # self.add_history(command,res)
        self.display(res)
    
    def Start(self):
        self.location = self.location_conditions.loc[0,"location"]
        self.end_of_game = False
        self.look_around()

    def chat_to(self,npc_id:str,command:str):
        # 和NPC交谈，放在UI里面
        npc = self.available_npcs[npc_id]
        original_status = npc.status['status_ID']
        res_command,res = npc.chat(command)
        self.new_display_history.extend(npc.display_history)
        if res_command is None:
            return
        self.add_chat_history(npc_id,"\n".join([res_command,res]))
        if npc.status['status_ID']!=original_status:
            if npc.status['status_ID']==0:
                self.add_chat_history(npc_id,f"{npc.name}被玩家气走了")
                # 在UI里做，删除这个人的按钮
            if npc.status["status_ID"]==2:
                self.battle(2,npc) #战斗 由于玩家可能死亡，故而将chat history做进battle函数中
                self.add_chat_history(npc_id,f"玩家与{npc.name}战斗并获胜，{npc.name}死亡")
                self.history.append({"role":"system","content":f"玩家与{npc.name}战斗并获胜，{npc.name}死亡"})
            if npc.status["status_ID"]==3: #成功说服；解析persuation_result
                persuation_result = npc.persuation_result
                persuation_result_txt = npc.persuation_result_txt
                #persuation_result = persuation_result.split(";")
                npc.display(persuation_result_txt)
                self.add_chat_history(npc_id,f"玩家成功说服{npc.name}，{npc.name}告诉玩家{persuation_result_txt}")
                self.history.append({"role":"system","content":f"玩家成功说服{npc.name}，{npc.name}告诉玩家{persuation_result_txt}"})
                self.new_display_history.append(npc.display_history[-1]) #加入说完毕的提示词
                self.parse_action_cause(persuation_result)
        for k,v in npc.induce.items():
            if v[2]==1:
                v[2]=2
                self.parse_action_cause(v[1])
        if npc_id == "6":
            npc.special_patient += 1
            if npc.special_patient >= 5:
                self.goto("小仓库",is_induce=True)
        

    def save(self,s="save.pkl"):
        with open(s,"wb") as f:
            pickle.dump(self.__dict__,f)

    def load(self,s="save.pkl"):
        # 总的DataFrams不随读取变化
        df_locations = self.location_conditions
        df_events = self.events
        df_npc = self.npcs
        df_stage_end = self.stage_end
        df_prompt = self.df_prompt
        with open(s,"rb") as f:
            self.__dict__.update(pickle.load(f))
        self.function_dicts = {"interaction_att":self.interaction_att,"attack":self.attack,"goto":self.goto,"necromancy":self.necromancy} # self的对象改了，所以这里的dict需要重新初始化
        self.location_conditions=df_locations
        self.events=df_events
        self.npcs=df_npc
        self.stage_end=df_stage_end
        self.df_prompt=df_prompt
    def get_prompt(self): #当前给到用户的提示，不是给到模型的
        current_prompt = self.df_prompt[self.df_prompt["location"]==self.location].reset_index()
        res = self.added_presets[:]
        deep_res = []
        for i in current_prompt.index:
            Flag = True
            if not pd.isna(current_prompt.loc[i,"condition"]):
                conditions = current_prompt.loc[i,"condition"].split(";")
                for c in conditions:
                    if c:
                        ctype,cvalue=c.split(":")
                        if (ctype == "NPC" and cvalue not in self.available_npcs.keys())\
                        or (ctype in self.status.keys() and self.status[ctype]<int(cvalue))\
                        or (ctype == "object" and cvalue not in self.available_objects):
                            Flag = False
                            break
                        elif ctype == "object_status":
                            obj,sts = cvalue.split(",")
                            if obj not in self.available_objects or (obj in self.available_objects and self.object_status[self.available_object_id[obj]] != sts):
                                Flag = False
                                break
            if Flag:
                if current_prompt.loc[i,"deep"]:
                    deep_res.append(current_prompt.loc[i,"prompt"])
                else:
                    res.append(current_prompt.loc[i,"prompt"])
        return res,deep_res
class Character:
    def __init__(self,
                 **kwargs
                 ):
        self.status={
            "strength":kwargs['strength'],
            "sense":kwargs['sense'],
            "health":kwargs['health'],
            "patient":kwargs['patient'], #归零时进入战斗或者离开
            "status_ID":1 #1正常 2战斗 0死亡/离开 3被说服
        }
        self.id=kwargs['ID']
        self.name=kwargs['name']
        self.anger_condition=kwargs['anger_condition']
        self.prompt=kwargs['prompt']
        # 打个补丁：有些时候会显示自身作为AI的思考
        if not pd.isna(self.prompt):
            self.prompt+="\n请只输出自己扮演的NPC的回复，不要写解释或是自身思考的过程。"
        self.battle_able=kwargs['battle_able']
        self.appear=kwargs['appear']
        self.disappear=kwargs['disappear']
        self.disappear_cause=kwargs['disappear_cause']
        self.disappear_anger=kwargs['disappear_anger']
        self.necromancy=kwargs['necromancy']
        self.necromancy_dead=kwargs['necromancy_dead']
        self.necromancy_dead_cause=kwargs['necromancy_dead_cause']
        self.init_word=kwargs['init_word']
        self.presets=kwargs['presets']
        self.persuade_key=kwargs['persuade_key']
        self.persuade_value=kwargs['persuade_value']
        self.persuation_result=kwargs['persuation_result']
        self.persuation_result_txt=kwargs['persuation_result_txt']
        self.persuated_prompt=kwargs['persuated_prompt']
        self.CSS_class=kwargs['CSS_class']
        self.is_necromancy = False
        self.persuade_key_dict = {i.split(";")[1]:False for i in self.persuade_key.split("\n")} if not pd.isna(self.persuade_key) else {}
        self.current_command = ""
        self.profile_img = base64_to_PIL(profiles[kwargs['profile_img']])#PIL.Image.open(kwargs['profile_img'])
        self.tools = []
        self.display_history=[]
        if not pd.isna(self.anger_condition):
            self.tools = [
                {
                "type": "function",
                "function": {
                "name": "less_patient",
                "description": f"{self.anger_condition}时调用的函数"
                }
                }
                ]
        self.tool_prompt = self.build_tool_prompt()
        if pd.isna(self.prompt):
            self.history=[]
        else:
            self.history = [{"role": "system", "content": self.prompt+self.tool_prompt}]
        if not pd.isna(kwargs["ancestor"]):
            self.history = kwargs["ancestor"].history+[{"role": "system", "content": self.prompt+self.tool_prompt}]
        if not pd.isna(self.appear):
            self.display(self.appear)
        if not pd.isna(self.init_word):
            self.display(self.init_word)
            self.history.append({"role":"assistant","content": self.init_word})

        self.threshold = 0.8 #大于等于这个数值，会推进说服点
        self.hurt_img = self.image_hurt()
        self.critical_img = self.image_critical()

        self.special_patient = 0 # 另一种“patient”,对话轮数超过一定数量后自动推进。
        self.induce = {} # 会诱发事件的指令：指令:(system提示,诱发,状态机) 其中0代表为激活，1代表激活，2代表已经激发过

    def image_hurt(self):
        img_data = base64.b64decode(hurt_img)
        img_file = io.BytesIO(img_data)
        hurt_effact = PIL.Image.open(img_file)
        res = self.profile_img.copy().resize((50,50))
        res.paste(hurt_effact,mask=hurt_effact)
        return res
    
    def image_critical(self):
        img_data = base64.b64decode(critical_img)
        img_file = io.BytesIO(img_data)
        critical_effact = PIL.Image.open(img_file)
        res = self.profile_img.copy().resize((50,50))
        res.paste(critical_effact,mask=critical_effact)
        return res


    def build_tool_prompt(self):
        broadcast_prompt = "如果输入信息的角色是system，则代表系统记录的玩家行动或玩家与其他npc的对话或者系统对你的提示，不是玩家本人对你说话时的输入。"
        if pd.isna(self.anger_condition):
            return broadcast_prompt
        tool_prompt = f"""
        ## 工具

        你有以下的工具可以使用：
        
        ### 感到愤怒

        less_patient: {self.anger_condition}时调用此函数

        ## 注意调用函数时只需要看用户的“上一个输入”，不要将用户的所有历史信息作为调用函数的依据。
        """
        return "\n".join([i.strip() for i in tool_prompt.splitlines()]+[broadcast_prompt])
    
    def display(self,s):
        if pd.isna(s):
            return
        # 图片处理
        if "placeholder_for_img" in s:
            for i in re.findall("\<placeholder_for_img:(.*?)\>",s):
                print(i)
                rpl = ""
                if i in illustrations.keys():
                    rpl = f"<img style=\"height: 1000px;\" src=\"data:image/jpeg;base64,{illustrations[i]}\">"
                s = s.replace(f"<placeholder_for_img:{i}>",rpl)
        self.display_history.append({
            "role":"assistant",
            "content": s.replace("<|im_end|>","").replace("\n","<br>"),
            "mode": "typewriter",
            "class":self.CSS_class,
            'style':"",
            "timestamp": time.time()
        })

    def add_history(self,user,assistant):
        self.history.extend([{"role":"user","content": user},
            {"role":"assistant","content": assistant}])

    def necromancy_to(self):
        if self.status["status_ID"]==0:
            self.display(self.necromancy_dead)
        self.display(self.necromancy)
        self.is_necromancy = True
    

    def build_presets(self,elo): #由主函数调用，最终还是要加一个口才函数……
        res = []
        # induce:
        for k,_ in self.induce.items():
            res.append(k)

        # persuade_key：
        if not pd.isna(self.persuade_key):
            for s in self.persuade_key.split("\n"):
                condition,content = s.split(";")
                flag = True
                # condition由2部分组成：口才要求(elo_condition)以及信息要求(sp，代表是否进行过通灵术)
                condition = condition.split(",")
                if len(condition)>1 and not self.is_necromancy:
                    flag = False
                c = int(condition[0])
                if elo>=c and c>0:
                    content = f"【通过口才判定:{c}】"+content
                if elo<c:
                    flag = False

                if flag:
                    res.append(content)
        if not pd.isna(self.presets):
            for s in self.presets.split("\n"):
                if s[0:4]=="[sp]":
                    if self.is_necromancy:
                        res.append(s[4:])
                else:
                    res.append(s)
        return res

    def add_preset(self,s):
        if s not in self.persuade_key_dict.keys():
            self.persuade_key_dict[s] = False
            if pd.isna(self.persuade_key):
                self.persuade_key = "0;"+s
            else:
                self.persuade_key += "\n0;"+s

    def similarity_sentence(self,sentence_ipt,sentence_list):
        def get_embeddings(sentences):
            completion = similar_client.embeddings.create(
                model=similar_model,
                input=sentences,
                dimensions=1024,
                encoding_format="float"
            )
            return [i["embedding"] for i in json.loads(completion.model_dump_json())['data']]
        def cosine_similarity(vec1, vec2):
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2)
        encoded_input = get_embeddings([sentence_ipt]+sentence_list)
        similar_res = [cosine_similarity(encoded_input[0],j) for j in encoded_input[1:]]
        return similar_res
    
    def similarest_persuade(self,sentence):
        sentence_list = list(self.persuade_key_dict.keys())
        similar_res = self.similarity_sentence(sentence,sentence_list)
        print({i:j for i,j in zip(sentence_list,similar_res)})
        if max(similar_res)<self.threshold:
            return False
        else:
            res = sentence_list[np.argmax(similar_res)]
            if not self.persuade_key_dict[res]:
                self.persuade_key_dict[res] = True
                return True
            else:
                return False
            
    def similarest_induce(self,sentence):
        sentence_list = [i for i in list(self.induce.keys()) if self.induce[i][2]==0]
        similar_res = self.similarity_sentence(sentence,sentence_list)
        print({i:j for i,j in zip(sentence_list,similar_res)})
        if max(similar_res)<self.threshold:
            return
        else:
            res = sentence_list[np.argmax(similar_res)]
            if self.induce[res][2]==0:
                self.induce[res][2] = 1
                self.history+=[{"role":"system","content":self.induce[res][0]}] #新增system提示，以此作为后文回答的依据之一
            
    def chat(self,command):
        if self.status["status_ID"] == 0:
            self.display(f"{self.name}已经死亡")
            return None,None
        self.current_command = command
        if len([i for i in list(self.induce.keys()) if self.induce[i][2]==0])>0:
            self.similarest_induce(command)
                
        if self.id == "16":
            # 进入地宫前的一个插曲
            similar_res = self.similarity_sentence(command,["你有没有远程投影的东西引开他们？"])
            if max(similar_res)>=self.threshold:
                res = "我已经使用过用随身携带的小手电照了，但是他们似乎全都没反应，哪怕直接照射他们的眼睛也没有。我怀疑他们不靠视力或听觉。"
                self.display(res)
                return f"玩家:{command}",f"{self.name}:{res}"
        if not pd.isna(self.persuade_value) and not pd.isna(self.persuade_key) and self.persuade_value > 0 and self.status["status_ID"]!=3: #说服
            if self.similarest_persuade(command):
                self.persuade_value -= 1
                if "<p style=\"visibility:hidden\">(清醒)</p>" in self.name:
                    rpl = "<p style=\"visibility:hidden\">(清醒)</p>"
                    self.display(f"{self.name.replace(rpl,'')}产生了一丝动摇")
                else:
                    if self.id in {"12","2"}:
                        self.display(f"{self.name}产生了一丝动摇")
                # self.history.append({"role":"system","content": f"{self.name}感到了一丝动摇"})
                if self.persuade_value == 0:
                    #self.display(self.persuation_result_txt)
                    self.status["status_ID"] = 3
                    #更新prompt
                    if pd.isna(self.persuated_prompt):
                        self.persuated_prompt = self.prompt
                    self.history[0] = {"role":"system","content":self.persuated_prompt+self.tool_prompt}
                    self.prompt=self.persuated_prompt
        if len(self.tools)>0:
            completion = client.chat.completions.create(
                    model=model,
                    messages=self.history+[{"role": "user",  "content": command}],
                    tools=self.tools
                )
        else:
            completion = client.chat.completions.create(
                    model=model,
                    messages=self.history+[{"role": "user",  "content": command}]
                )
        res = completion.choices[0].message.content
        # 有函数调用的情况
        if completion.choices[0].message.tool_calls is not None:
            self.less_patient()
            return f"玩家:{command}",f"这使得{self.name}感到一丝愤怒"
        self.add_history(command,res)
        self.display(res)
        return f"玩家:{command}",f"{self.name}:{res}"
        
    def less_patient(self):
        self.status["patient"] -= 1
        self.display(f"{self.name}感到了一丝愤怒")
        if self.status["patient"] == 0:
            if self.battle_able==1:
                self.status["status_ID"] = 2
            else:
                self.die()
        else:
            self.history[0] = {"role": "system", "content": self.prompt} #暂时不用任何函数
            # self.history.extend([{"role":"user","content": self.current_command},
            # {"role":"system","content": f"{self.name}感到了一丝愤怒"}])

            completion = client.chat.completions.create(
                model=model,
                messages=self.history#+[{"role":"system","content": f"{self.name}感到了一丝愤怒"}]
            )
            res = completion.choices[0].message.content
            
            self.display(res)
            # self.history.append({"role":"assistant","content": res}) 同一句话多次都可以激怒对方
            self.history[0] = {"role": "system", "content": self.prompt+self.tool_prompt} #修改回来，下次还得怒
    def die(self,how="anger"):
        if how=="anger": #被气走了
            self.display(self.disappear_anger)
        self.status["status_ID"] = 0 #不再可交互
        self.profile_img = self.profile_img.convert("L")

class GameUI(Game):
    def __init__(self,df_locations,df_events,df_npc,df_stage_end,df_prompt):
        super().__init__(df_locations,df_events,df_npc,df_stage_end,df_prompt)
        self.chat_history = []
        self.display_mode = "typewriter"  # 可选 "instant" 或 "typewriter"
        self.html_content=""
        self.system_profile = base64_to_PIL(profiles['system'])
        self.hurt_img = self.image_hurt()
        self.critical_img = self.image_critical()
        self.chosen_npc = "system"
        self.is_activate = False
        self.init_text = """
        <div class=\"system-info\">请在右侧进行加点完成后开始游戏<br>
        【体质】<br>
        昭示肉身承载天地戾气的器量，战斗中亦为攻防依据。<br>
        <p class=\"trait-warning\">但是哪怕能空手搏虎，也有可能亡于闺阁绣针</p>

        【感知】<br>
        五感通达，眼力入微；也能在战斗中观察破绽与走势。<br>
        <p class=\"trait-warning\">但是哪怕真的能感知万物，无法改变又只会陡增悲伤</p>

        【口才】<br>
        七窍玲珑心，洞察对方喜怒；能够知晓如何说服对方。<br>
        <p class=\"trait-warning\">但是如果真的说到了对方的痛点，哪怕无需口才，亦能成事</p>
        <p class=\"trait-warning\">祸从口出，说客终南捷径，往往通向五马分尸柱。</p></div>
        """
        
        self.init_text = "".join([i.strip() for i in self.init_text.split("\n")])
        
        self.instant = False #战斗是否立即结束
        self.quiet = False #是否禁止背景音乐
        self.can_change_npc = True #对话显示时，不可以切换NPC

        scripts = """
        async () => {
            globalThis.copy = (idx) => {
                s = document.getElementById(\"user_prompt_\"+idx).innerText;
                navigator.clipboard.writeText(s);
                d = document.querySelector(\"#user_prompt_\"+idx+\" > img\");
                d.src = \"data:image/jpeg;base64,"""+checked+"""\";
                setTimeout(function(){
                    d = document.querySelector(\"#user_prompt_\"+idx+\" > img\");
                    d.src = \"data:image/jpeg;base64,"""+copy_icon+"""\";
                },300);
            };
            globalThis.instant = () => {
                btn = document.getElementById(\"instant_gradio\");
                btn.click();
            }
            globalThis.quiet = () => {
                btn = document.getElementById(\"quiet_btn\");
                btn.click();
            }
        }
        """

        with gr.Blocks(head="<link rel=\"stylesheet\" href=\"https://cdn.bootcdn.net/ajax/libs/lxgw-wenkai-screen-webfont/1.7.0/style.min.css\">\
                       <link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Ma+Shan+Zheng&family=ZCOOL+QingKe+HuangYou&family=ZCOOL+KuaiLe&family=Noto+Serif+SC:wght@500;700&ZCOOL+KuaiLe&family=ZCOOL+XiaoWei&family=Oxanium&family=Share+Tech+Mono&family=ZCOOL+QingKe+HuangYou&family=Noto+Sans+SC&family=Hanyi+Senty+Marshmallow&display=swap\">",css=css,js=scripts) as self.demo:
            #self.game_state = gr.State(self)
            
            with gr.Row():
                # 左侧NPC选择
                with gr.Column(scale=1,min_width=100):
                    self.NPC_list = [(gr.Image(None,width="50px",height="50px",elem_id=f"npc_{i}"
                                               ,visible=False,container=False,show_label=False,show_download_button=False,show_fullscreen_button=False,show_share_button=False)
                                     ,gr.HTML("",elem_id=f"npc_label_{i}",visible=False)) for i in range(6)] #最多5个NPC
                # 中间聊天区
                with gr.Column(scale=100):
                    self.chat_display = gr.Chatbot(value=[[None,self.init_text]],elem_classes="HTML_container",show_label=False,min_height=600)
                    with gr.Row():
                        self.prompt_display = gr.HTML(padding=0)
                        self.prompt_display_deep = gr.HTML(padding=0)
                    self.prompt_button = gr.Button("提示",interactive=False)
                    self.deep_prompt_button = gr.Button("更多提示",visible=False)
                    self.message_input = gr.Textbox(label=None, placeholder="输入指令后按回车或发送按钮",interactive=False,elem_id="send_msg",container=False)
                    self.send_btn = gr.Button("发送",interactive=False)
                    self.start_btn = gr.Button("开始游戏",interactive=False)

                # 右侧角色状态、存读档
                with gr.Column(scale=3,elem_classes="current_status",min_width=300):
                    self.health_display=gr.HTML(f"<p>生命：{self.status['health']}</p>")
                    with gr.Row(variant="compact"):
                        self.strength_display=gr.HTML(f"体质：{self.status['strength']}",elem_classes="status_display")
                        self.strength_button_plus = gr.Button("+",elem_id="strength_button_plus",elem_classes="plus_button",min_width=10)
                        self.strength_button_minus = gr.Button("-",elem_id="strength_button_minus",elem_classes="minus_button",min_width=10,interactive=False)
                    with gr.Row(variant="compact"):
                        self.sense_display=gr.HTML(f"感知：{self.status['sense']}",elem_classes="status_display")
                        self.sense_button_plus = gr.Button("+",elem_id="sense_button_plus",elem_classes="plus_button",min_width=10)
                        self.sense_button_minus = gr.Button("-",elem_id="sense_button_minus",elem_classes="minus_button",min_width=10,interactive=False)
                    with gr.Row(variant="compact"):
                        self.eloquent_display=gr.HTML(f"口才：{self.status['eloquent']}",elem_classes="status_display")
                        self.eloquent_button_plus = gr.Button("+",elem_id="eloquent_button_plus",elem_classes="plus_button",min_width=10)
                        self.eloquent_button_minus = gr.Button("-",elem_id="eloquent_button_minus",elem_classes="minus_button",min_width=10,interactive=False)
                    self.point_display = gr.HTML(f"剩余点数：{self.points}")
                    self.instant_btn = gr.Button("战斗快进",elem_id="instant_gradio",visible=False)
                    self.bgm_audio = gr.Audio(visible=False)
                    self.hint = gr.HTML("默认保存路径为py文件/notebook文件根目录；会在目录下生成一个名为\"save_AI_ImmersiveSimGame.pkl\"的文件，多次保存会覆盖，还请留意。")
                    self.save_path = gr.Textbox(label="保存目录", placeholder="请输入保存目录",value=os.getcwd())
                    self.save_button = gr.Button("保存",interactive=False)
                    self.load_button = gr.Button("读取")
                    self.bgm = gr.Audio(value=None,type="numpy",visible=False)
                    self.attack_voice = gr.Audio(value=None,type="numpy",visible=False)

                    self.quiet_btn = gr.Button(visible=False,elem_id="quiet_btn")
                    self.quiet_icon = gr.HTML(f'<img src="data:image/png;base64,{unquiet}" onclick=quiet()>',elem_id="quiet_icon")

            self.add_buttons = [self.strength_button_plus,self.sense_button_plus,self.eloquent_button_plus]
            self.minus_buttons = [self.strength_button_minus,self.sense_button_minus,self.eloquent_button_minus]
            self.status_labels  = [self.health_display,self.strength_display,self.sense_display,self.eloquent_display,self.point_display,self.start_btn]
            
            self.update_widgets = [self.chat_display,self.message_input,self.send_btn,self.start_btn]+[i[0] for i in self.NPC_list]+[i[1] for i in self.NPC_list]+\
                [self.health_display,self.strength_display,self.sense_display,self.eloquent_display,self.point_display,self.prompt_button,self.deep_prompt_button,self.prompt_display,self.prompt_display_deep]+\
                    self.add_buttons+self.minus_buttons+[self.save_button,self.load_button,self.instant_btn]+[self.bgm,self.attack_voice,self.quiet_icon]
            update_keys = ["chat_display","message_input","send_btn","start_btn"]+[i[0].elem_id for i in self.NPC_list]+[i[1].elem_id for i in self.NPC_list]+\
                    ["health_display","strength_display","sense_display","eloquent_display","points_display","prompt_button","deep_prompt_button","prompt_display","prompt_display_deep"]+\
                    ["strength_button_plus","sense_button_plus","eloquent_button_plus"]+["strength_button_minus","sense_button_minus","eloquent_button_minus","save_button","load_button","instant_button"]+\
                    ["bgm","attack_voice","quiet_icon"]
            self.update_dict = {update_keys[i]:self.update_widgets[i] for i in range(len(self.update_widgets))}
            #上面这3个变量是用来做update_outputs的，每个函数想要什么更新什么组件就自己调用update_outputs函数，它会自动按顺序输出每个组件的更新值
            self.send_btn.click(
                fn=self.send_message,
                inputs=[self.message_input],
                outputs = self.update_widgets,
                show_progress=False
            ).then(
                fn=self.update_new_chat,
                outputs=self.update_widgets,
                show_progress=False
            )
            self.message_input.submit(
                fn=self.send_message,
                inputs=[self.message_input],
                outputs = self.update_widgets,
                show_progress=False
            ).then(
                fn=self.update_new_chat,
                outputs=self.update_widgets,
                show_progress=False
            )
            self.start_btn.click(
                fn=self.Start,
                outputs=self.update_widgets,
                show_progress=False
            ).then(
                fn=self.update_new_chat,
                outputs=self.update_widgets,
                show_progress=False
            )
            
            self.strength_button_plus.click(
                fn=self.add_strength,
                outputs = self.add_buttons+self.minus_buttons+self.status_labels,
                show_progress=False
            )
            self.strength_button_minus.click(
                fn=self.minus_strength,
                outputs = self.add_buttons+self.minus_buttons+self.status_labels,
                show_progress=False
            )

            self.sense_button_plus.click(
                fn=self.add_sense,
                outputs = self.add_buttons+self.minus_buttons+self.status_labels,
                show_progress=False
            )
            self.sense_button_minus.click(
                fn=self.minus_sense,
                outputs = self.add_buttons+self.minus_buttons+self.status_labels,
                show_progress=False
            )

            self.eloquent_button_plus.click(
                fn=self.add_elo,
                outputs = self.add_buttons+self.minus_buttons+self.status_labels,
                show_progress=False
            )
            self.eloquent_button_minus.click(
                fn=self.minus_elo,
                outputs = self.add_buttons+self.minus_buttons+self.status_labels,
                show_progress=False
            )

            for (i,j) in self.NPC_list:
                i.select(
                    fn=self.change_selected_NPC,
                    inputs = j,
                    outputs=self.update_widgets,
                    show_progress=False
                )
            
            self.save_button.click(
                fn=self.save,
                inputs=self.save_path,
                outputs=self.update_widgets,
                show_progress=False
                )

            self.load_button.click(
                fn=self.load,
                inputs=self.save_path,
                outputs=self.update_widgets,
                show_progress=False,
                ).then(
                    fn=self.update_bgm,
                    outputs=[self.bgm]
                )

            self.prompt_button.click(
                fn=self.upate_user_prompts,
                outputs=self.update_widgets,
                show_progress=False
            )
            self.deep_prompt_button.click(
                fn=self.update_deep_prompts,
                outputs=self.update_widgets,
                show_progress=False
            )

            self.instant_btn.click(
                fn = self.set_instant
            )

            self.quiet_btn.click(
                fn=self.set_quiet,
                outputs=[self.quiet_icon],
                show_progress=False
            ).then(
                fn=self.update_bgm,
                outputs=[self.bgm],
                show_progress=False
            )

        self.demo.launch()
        
    def set_instant(self):
        self.instant=True
    
    def set_quiet(self):
        if not self.quiet:
            self.quiet = True
            return gr.update(value=f'<img src="data:image/png;base64,{quiet}" onclick=quiet()>')
        else:
            self.quiet = False
            return gr.update(value=f'<img src="data:image/png;base64,{unquiet}" onclick=quiet()>')

    def update_bgm(self):
        if not self.current_bgm:
            return gr.update(value=None)
        if self.is_changed_bgm:
            self.is_changed_bgm = False
        return gr.update(value=bgms[self.current_bgm],type="numpy",autoplay=(not self.quiet),loop=True)

    def image_hurt(self):
        img_data = base64.b64decode(hurt_img)
        img_file = io.BytesIO(img_data)
        hurt_effect = PIL.Image.open(img_file)
        res = self.system_profile.copy().resize((50,50))
        res.paste(hurt_effect,mask=hurt_effect)
        return res
    
    def image_critical(self):
        img_data = base64.b64decode(critical_img)
        img_file = io.BytesIO(img_data)
        critical_effect = PIL.Image.open(img_file)
        res = self.system_profile.copy().resize((50,50))
        res.paste(critical_effect,mask=critical_effect)
        return res

    def display(self,text,elem_class="system-info",style="",mode="typewriter",is_battle=0,battle_info={}):
        if pd.isna(text):
            return
        # 图片处理
        if "placeholder_for_img" in text:
            for i in re.findall("\<placeholder_for_img:(.*?)\>",text):
                print(i)
                rpl = ""
                if i in illustrations.keys():
                    rpl = f"<img style=\"height: 1000px;\" src=\"data:image/jpeg;base64,{illustrations[i]}\">"
                text = text.replace(f"<placeholder_for_img:{i}>",rpl)
        if self.location in ["深层意识","恐怖分子基地","处决场","记忆圣所"] and elem_class == "system-info":
            elem_class = "system-info-snake"
        if is_battle == 2: #结束战斗
            self.batting = False
            return
        if is_battle == 1: #开始战斗
            self.batting = True
        if self.batting:
            battle_info.update({"text":text})
            self.battle_log.append(battle_info)
            self.new_display_history.append({
                "role":"assistant",
                "content": "",
                "mode": "placeholder_for_battle",
                "class":"",
                "style":'',
                "timestamp": time.time()
            })
        else:
            self.new_display_history.append({
                "role":"assistant",
                "content": text.replace("<|im_end|>","").replace("\n","<br>"),
                "mode": mode,
                "class":elem_class,
                "style":'"'+style+'"',
                "timestamp": time.time()
            })
        
    def add_strength(self):
        self.status["strength"] += 1
        self.points-=1
        self.status["health"] += 100
        return self.adjust_point()
    def minus_strength(self):
        self.status["strength"] -= 1
        self.points+=1
        self.status["health"] -= 100
        return self.adjust_point()
    
    def add_sense(self):
        self.status["sense"] += 1
        self.points-=1
        return self.adjust_point()
    def minus_sense(self):
        self.status["sense"] -= 1
        self.points+=1
        return self.adjust_point()
    
    def add_elo(self):
        self.status["eloquent"] += 1
        self.points-=1
        return self.adjust_point()
    def minus_elo(self):
        self.status["eloquent"] -= 1
        self.points+=1
        return self.adjust_point()
    
    def adjust_point(self):
        res = []
        # 加号
        res.extend([gr.update(value="+",interactive=self.points > 0)]*3)
        # 减号
        res.extend([gr.update(value="-",interactive=self.status["strength"] > self.min_sp),
                    gr.update(value="-",interactive=self.status["sense"] > self.min_sp),
                    gr.update(value="-",interactive=self.status["eloquent"] > self.min_elo)])
        # 标签
        res.extend([gr.update(value=f"生命：{self.status['health']}"),
                    gr.update(value=f"体质：{self.status['strength']}"),
                    gr.update(value=f"感知：{self.status['sense']}"),
                    gr.update(value=f"口才：{self.status['eloquent']}"),
                    gr.update(value=f"剩余点数：{self.points}")])
        res.append(gr.update(interactive=self.points==0))
        return res

    def update_outputs(self,**kwargs):
        update_dict = {i:gr.update() for i in self.update_widgets}
        for k,v in kwargs.items():
            update_dict[self.update_dict[k]] = v
        if len(self.battle_log)==0: #如果不是battle，就直接更新生命
            update_dict[self.update_dict["health_display"]]=gr.update(value=f"生命：{max(self.status['health'],0)}")
        update_dict[self.update_dict["strength_display"]]=gr.update(value=f"体质：{self.status['strength']}")
        update_dict[self.update_dict["sense_display"]]=gr.update(value=f"感知：{self.status['sense']}")
        update_dict[self.update_dict["eloquent_display"]]=gr.update(value=f"口才：{self.status['eloquent']}")
        update_dict[self.update_dict["points_display"]]=gr.update(value=f"剩余点数：{self.points}")
        if self.location in ["深层意识","恐怖分子基地","处决场","记忆圣所"]:
            update_dict[self.update_dict["strength_display"]]=gr.update(value=f"体质：15")
            update_dict[self.update_dict["sense_display"]]=gr.update(value=f"感知：15")
            update_dict[self.update_dict["eloquent_display"]]=gr.update(value=f"口才：15")
        elif self.location == "梵脉":
            update_dict[self.update_dict["health_display"]]=gr.update(value=f"生命：0")
            update_dict[self.update_dict["strength_display"]]=gr.update(value=f"体质：0")
            update_dict[self.update_dict["sense_display"]]=gr.update(value=f"感知：0")
            update_dict[self.update_dict["eloquent_display"]]=gr.update(value=f"口才：0")
        # if self.is_changed_bgm:
        #         # 需要先停止现在的音乐
        #         update_dict[self.update_dict["bgm"]]=gr.update(value=None,interactive=False)
        #     else:
        #         update_dict[self.update_dict["bgm"]]=self.update_bgm()
        #         self.is_changed_bgm = False
        if self.end_of_game == True: #结局了，不能再操作某些特定的控件了
            update_dict[self.update_dict["message_input"]]=gr.update(interactive=False)
            update_dict[self.update_dict["send_btn"]]=gr.update(value="发送",interactive=False)
            update_dict[self.update_dict["save_button"]]=gr.update(interactive=False)
            update_dict[self.update_dict["start_btn"]]=gr.update(value="重新开始",interactive=True)
        if self.quiet:
            update_dict[self.update_dict["quiet_icon"]] = gr.update(value=f'<img src="data:image/png;base64,{quiet}" onclick=quiet()>')
        else:
            update_dict[self.update_dict["quiet_icon"]] = gr.update(value=f'<img src="data:image/png;base64,{unquiet}" onclick=quiet()>')
        return [update_dict[i] for i in self.update_widgets]

    def upate_user_prompts(self):
        deep_prompt = []
        prompts_for_user = []
        res = {}
        res["prompt_display_deep"]=None

        res["prompt_button"]=gr.update(interactive=False)
        if self.chosen_npc == "system": #系统提示
            prompts_for_user,deep_prompt = self.get_prompt()
            prompts_for_user = [f"<p class=\"user_prompt\">{i}" for i in prompts_for_user]
            res["deep_prompt_button"] = gr.update(interactive=True,visible=True)
        else:
            prompts_for_user = self.available_npcs[self.chosen_npc].build_presets(self.status["eloquent"])
            prompts_for_user = [f"<p id=\"user_prompt_{i}\" class=\"user_prompt\">{x}<img class=\"copy-icon\" onclick=\"copy({i})\" src=\"data:image/jpeg;base64,{copy_icon}\">" for i,x in enumerate(prompts_for_user)]
            res["deep_prompt_button"] = gr.update(visible=False)
        prompts_for_user = "".join(prompts_for_user)
        
        res["prompt_display"] = gr.update(value = prompts_for_user)
        res.update(self.update_npc_list())
        return self.update_outputs(chat_display=self.get_historical_chat(),**res)

    def update_deep_prompts(self):
        res = {}
        res["deep_prompt_button"]=gr.update(interactive=False)
        res.update(self.update_npc_list())
        prompts_for_user,deep_prompt = self.get_prompt()
        prompts_for_user = "".join([f"<p class=\"user_prompt\">{i}" for i in prompts_for_user])
        deep_prompt = "".join([f"<p class=\"user_prompt\">{i}" for i in deep_prompt])
        res["prompt_display"]=gr.update(value = prompts_for_user)
        res["prompt_display_deep"]=gr.update(value=deep_prompt)
        return self.update_outputs(chat_display=self.get_historical_chat(),**res)

    def Start(self):
        visible_components = ["strength_button_plus","sense_button_plus","eloquent_button_plus","strength_button_minus","sense_button_minus","eloquent_button_minus","points_display"]
        if not self.is_activate: #开始游戏
            self.chat_history = []
            self.new_display_history = []
            self.is_activate = True
            self.location = self.location_conditions.loc[0,"location"]
            self.look_around()
            res = {i:gr.update(visible=False) for i in visible_components}
            res["chat_display"]=self.get_historical_chat()
            res["send_btn"]=gr.update(interactive=False)
            res["prompt_button"]=gr.update(interactive=False)
            res["start_btn"]=gr.update(value="重新开始",interactive=False)
            res["message_input"]=gr.update(interactive=False)
            res["save_button"]=gr.update(interactive=False)
            res["load_button"]=gr.update(interactive=False)
            res["bgm"] = self.update_bgm()
            res.update(self.update_npc_list())
            self.end_of_game = False
            return self.update_outputs(**res)
        else: #重新开始
            super().__init__(df_locations,df_events,df_npc,df_stage_end,df_prompt)
            self.current_bgm=None
            self.is_changed_bgm = True
            self.chat_history = [{
            "role":"assistant",
            "content": self.init_text,
            "mode": "None",
            "class": None,
            "style":None,
            "timestamp": time.time()
            }]
            self.new_display_history = []
            self.is_activate = False
            res = {i:gr.update(visible=True) for i in visible_components}
            for i in ["strength_button_minus","sense_button_minus","eloquent_button_minus"]:
                res[i] = gr.update(visible=True,interactive=False)
            for i in ["strength_button_plus","sense_button_plus","eloquent_button_plus"]:
                res[i] = gr.update(visible=True,interactive=True)
            res["chat_display"]=self.get_historical_chat()
            res["send_btn"]=gr.update(interactive=False)
            res["prompt_button"]=gr.update(interactive=False)
            res["start_btn"]=gr.update(value="开始游戏",interactive=False)
            res["message_input"]=gr.update(interactive=False)
            res["save_button"]=gr.update(interactive=False)
            res["bgm"]=gr.update(visible=False,interactive=False)
            self.end_of_game = False
            res.update(self.update_npc_list())
            return self.update_outputs(**res)

    def get_historical_chat(self):
        res = []
        for msg in self.chat_history:
            if msg['mode'] == 'placeholder_for_battle':
                continue
            if msg["role"] == "user":
                res.append([f"<p style={msg['style']}>"+ msg["content"]+'</p>',None])
            else:
                res.append([None,f"<p class=\"{msg['class']}\" style={msg['style']}>"+ msg["content"].replace("<p","<span").replace('</p>',"</span>")+'</p>'])
        res.append([None,""])
        return res

    def update_new_chat(self):
        if self.is_changed_bgm:
            yield self.update_outputs(bgm=self.update_bgm())
        if not self.is_activate:
            yield self.update_outputs(prompt_display=gr.update(value=None),prompt_display_deep=gr.update(value=None),
                                  prompt_button=gr.update(interactive=False),deep_prompt_button=gr.update(visible=False),message_input=gr.update(interactive=False),
                                  save_button=gr.update(interactive=False),load_button=gr.update(interactive=True),
                                  send_btn = gr.update(value="发送",interactive=False),start_btn=gr.update(interactive=False))
        else:
            res = []
            self.instant=False
            self.can_change_npc = False #暂时不能切换NPC
            for msg in self.chat_history:
                if msg['mode'] == 'placeholder_for_battle':
                    continue
                if msg["role"] == "user":
                    res.append([f"<p style={msg['style']}>"+ msg["content"]+'</p>',None])
                else:
                    res.append([None,f"<p class=\"{msg['class']}\" style={msg['style']}>"+ msg["content"].replace("<p","<span").replace('</p>',"</span>")+'</p>'])
            
            for msg in self.new_display_history:
                # 对于battle做特殊处理
                if len(self.battle_log)>0 and "mode" in msg and msg["mode"]=="placeholder_for_battle":
                    sleep_sec = 0.5
                    for i in self.line_by_line_battle(self.battle_log,res):
                        if self.instant:
                            sleep_sec=0
                        time.sleep(sleep_sec)
                        if self.quiet or self.instant:
                            i["attack_voice"] = gr.update()
                        yield self.update_outputs(**i)
                    self.battle_log = []
                elif msg['mode'] == 'placeholder_for_battle' and self.battle_log == []:
                    continue
                elif "mode" in msg and msg["mode"]=="LineByLine":
                    sleep_sec = 1
                    for i in self.line_by_line(msg,res):
                        time.sleep(sleep_sec)
                        yield self.update_outputs(chat_display=i)
                else:
                    sleep_sec = 0.01
                    for i in self.typewriter(msg,res):
                        time.sleep(sleep_sec)
                        yield self.update_outputs(chat_display=i)
            self.chat_history.extend(self.new_display_history)
            self.new_display_history = []
            self.can_change_npc = True #可以切换了
            yield self.update_outputs(prompt_display=gr.update(value=None),prompt_display_deep=gr.update(value=None),
                                    prompt_button=gr.update(interactive=True),deep_prompt_button=gr.update(visible=True),message_input=gr.update(interactive=True),
                                    save_button=gr.update(interactive=True),load_button=gr.update(interactive=True),
                                    send_btn = gr.update(value="发送",interactive=True),start_btn=gr.update(interactive=True))
    
    def line_by_line_battle(self,msg,res):
        res.append([None,f"<div class=\"battle\"></div>"])
        # res_outputs = []
        current_output = self.update_npc_list()
        for i,line in enumerate(msg):
            if line["text"] in ("begin","end"):
                continue
            res[-1][-1] = res[-1][-1].replace("</div>","")
            res[-1][-1]+="<p>"+line["text"]+"</p></div>"
            current_output["instant_button"]=gr.update(visible=True)
            if i==len(msg)-1:
                current_output["instant_button"]=gr.update(visible=False)
                res[-1][-1] = res[-1][1].replace("battle","battle-finished")
                self.chat_history.append({
                    'role': 'assistant',
                    'content': ("".join(res[-1][-1])),
                    'mode': 'LineByLine',
                    'class': '',
                    'style': '',
                    'timestamp': time.time()
                })
            current_output["chat_display"] = res
            if line["result"] != "missing":
                current_bgm = bgms["critical"] if line["result"] == "critical" else bgms["attack"]
                current_output["attack_voice"] = gr.update(value=current_bgm,type="numpy",autoplay=(not self.quiet))
                if line["target"] == "self":
                    current_output["health_display"] = gr.update(value=f"生命：{max(line['remaining_hp'],0)}")
                else:
                    # 根据id找组件
                    tgt_label = None
                    for i,(k,v) in enumerate(self.available_npcs.items()):
                        if k==line["target"]:
                            current_output[f"npc_label_{i+1}"] = gr.update(value=f"<div style=\"visibility:hidden;height:0;\">{k}</div><div style=\"text-align:center;\">{v.name}</div>\
                                                 <div class=\"npc_info\">命/体/感<br>{max(line['remaining_hp'],0)}/{v.status['strength']}/{v.status['sense']}</div>")
                            break
                
            # 特效会一直留在头像上，暂时不做了
            # if line["result"] != "missing" and i!=len(msg)-1:
            #     if line["target"]=="self":
            #         update_tgt = "npc_0"
            #         sp = self.critical_img if line["result"] == "critical" else self.hurt_img
            #     else:
            #         # 根据npc ID搜索控件的ID
            #         for j,(k,v) in enumerate(self.available_npcs.items()):
            #             if line["target"] == k:
            #                 update_tgt = f"npc_{j+1}"
            #                 sp = v.critical_img if line["result"] == "critical" else v.hurt_img
            #     current_output[update_tgt]=sp
            # res_outputs.append(current_output)
            yield current_output

    def line_by_line(self,msg,res):
        res.append([None,f"<div class=\"{msg['class']}\" style={msg['style']}></div>"])
        fadeIn_str = "fadeIn-system-snake" if msg['class']=="system-info-snake" else "fadeIn"
        rpl_res = " style=\"color:#C0D8C0;\"" if fadeIn_str == "fadeIn-system-snake" else ""
        for i,line in enumerate(msg["content"].split("<br>")):
            res[-1][-1] = res[-1][-1].replace("</div>","").replace(f" class=\"{fadeIn_str}\"",rpl_res).replace("-fadeIn","") #特殊人物的fadeIn取消
            res[-1][-1]+=f"<p class=\"{fadeIn_str}\">"+line+"</p></div>"
            #time.sleep(1)
            yield res

    def typewriter(self,msg,res):
        res.append([None,f"<p class=\"{msg['class']}\" style={msg['style']}></p>"])
        for i,part in enumerate(re.split('(<[^>]+>)',msg["content"])):
            if part:
                if part[0]=="<" and part[-1]==">":
                    res[-1][-1] = res[-1][-1].replace("</p>","")
                    res[-1][-1]+=part+"</p>"
                    yield res
                else:
                    for line in part:
                        res[-1][-1] = res[-1][-1].replace("</p>","")
                        res[-1][-1]+=line+"</p>"
                        yield res
            
    def change_selected_NPC(self,label):
        if not self.can_change_npc:
            return self.update_outputs()
        self.chosen_npc = re.findall('<div style=\"visibility:hidden;height:0;\">(.*?)</div>',label)[0]
        
        return self.update_outputs(chat_display=self.get_historical_chat(),
                                   prompt_display=gr.update(value=None),prompt_display_deep=gr.update(value=None),message_input=gr.update(value=None),
                                   prompt_button=gr.update(interactive=True),deep_prompt_button=gr.update(visible=False),
                                   **self.update_npc_list())

    def update_npc_list(self):
        res = {"npc_0":gr.update(visible=True,value=self.system_profile,container=False,elem_classes="unchosen_npc"),
               "npc_label_0":gr.update(visible=True,value="<div style=\"visibility:hidden;height:0;\">system</div><div style=\"text-align:center;\">系统</div>")}
        if self.chosen_npc == "system":
            res["npc_0"]=gr.update(visible=True,value=self.system_profile,container=True,elem_classes="chosen_npc")
        for i,(k,v) in enumerate(self.available_npcs.items()):
            if self.chosen_npc == k:
                res[f"npc_{i+1}"] = gr.update(visible=True,value=v.profile_img,elem_classes="chosen_npc",container=True)
            else:
                res[f"npc_{i+1}"] = gr.update(visible=True,value=v.profile_img,container=False,elem_classes="unchosen_npc")
            res[f"npc_label_{i+1}"] = gr.update(visible=True,value=f"<div style=\"visibility:hidden;height:0;\">{k}</div><div style=\"text-align:center;\">{v.name}</div>\
                                                <div class=\"npc_info\">命/体/感<br>{max(v.status['health'],0)}/{v.status['strength']}/{v.status['sense']}</div>")
        # 对于没有NPC的栏位，设置隐形
        for i in range(len(self.available_npcs.keys())+1,6): #最多5个角色
            res[f"npc_{i}"] = gr.update(visible=False)
            res[f"npc_label_{i}"] = gr.update(visible=False)
        return res
    
    def send_message(self,message):
        # 先限定所有的组件不能动
        yield self.update_outputs(prompt_display=gr.update(value=None),prompt_display_deep=gr.update(value=None),
                                  prompt_button=gr.update(interactive=False),deep_prompt_button=gr.update(visible=False),message_input=gr.update(value=None,interactive=False),
                                  save_button=gr.update(interactive=False),load_button=gr.update(interactive=False),
                                  send_btn = gr.update(value="响应中...",interactive=False),start_btn=gr.update(interactive=False))
        self.can_change_npc = False # 暂时不能切换NPC
        #清除所有NPC的display_history
        for _,npc in self.available_npcs.items():
            npc.display_history = []
        self.chat_history.append({
            "role":"user",
            "content": message,
            "mode": "None",
            "style":"font-family: KaiTi",
            "timestamp": time.time()
        })
        if self.chosen_npc == 'system':
            self.give_command(message)
        else:
            self.chat_to(self.chosen_npc,message)
        if self.is_changed_bgm:
            yield self.update_outputs(chat_display=self.get_historical_chat(),
                                   prompt_display=gr.update(value=None),prompt_display_deep=gr.update(value=None),bgm=gr.update(value=None,interactive=False),
                                   prompt_button=gr.update(interactive=False),deep_prompt_button=gr.update(visible=False),message_input=gr.update(value=None),
                                   **self.update_npc_list())
        else:
            yield self.update_outputs(chat_display=self.get_historical_chat(),
                                    prompt_display=gr.update(value=None),prompt_display_deep=gr.update(value=None),
                                    prompt_button=gr.update(interactive=False),deep_prompt_button=gr.update(visible=False),message_input=gr.update(value=None),
                                    **self.update_npc_list())
    
    def save(self,file_path):
        not_save_set={'function_dicts','NPC_list','update_widgets','update_dict','add_buttons','minus_buttons','status_labels'}
        save_values = {k:v for k,v in self.__dict__.items() if not callable(v) and 'gradio.components' not in str(type(v)) and k not in not_save_set}
        try:
            if not (os.path.exists(file_path) and os.path.isdir(file_path)):
                os.mkdir(file_path)                
            with open(os.path.join(file_path,"save_AI_ImmersiveSimGame.pkl"),"wb") as f:
                pickle.dump(save_values,f)
            self.chat_history.append({
                "role":"assistant",
                "content": "保存成功",
                "mode": "None",
                "style":'font-family: KaiTi',
                "class":"system-info",
                "timestamp": time.time()
            })
            return self.update_outputs(chat_display=self.get_historical_chat(),**self.update_npc_list())
        except:
            self.chat_history.append({
                "role":"assistant",
                "content": "保存失败，请检查路径是否正确",
                "mode": "None",
                "style":'font-family: KaiTi',
                "class":"system-info",
                "timestamp": time.time()
            })
            return self.update_outputs(chat_display=self.get_historical_chat(),**self.update_npc_list())
    
    def load(self,file_path):
        try:
            
            df_locations = self.location_conditions
            df_events = self.events
            df_npc = self.npcs
            df_stage_end = self.stage_end
            df_prompt = self.df_prompt
            with open(os.path.join(file_path,"save_AI_ImmersiveSimGame.pkl"),"rb") as f:
                saved_values=pickle.load(f)
            self.__dict__.update(saved_values)
            self.function_dicts = {"interaction_att":self.interaction_att,"attack":self.attack,"goto":self.goto,"necromancy":self.necromancy} # self的对象改了，所以这里的dict需要重新初始化
            self.location_conditions=df_locations
            self.events=df_events
            self.npcs=df_npc
            self.stage_end=df_stage_end
            self.df_prompt=df_prompt
        except:
            self.chat_history.append({
                "role":"assistant",
                "content": "读取游戏失败，请检查路径是否正确",
                "mode": "None",
                "style":'font-family: KaiTi',
                "class":"system-info",
                "timestamp": time.time()
            })
        self.chosen_npc = "system"
        visible_components = ["strength_button_plus","sense_button_plus","eloquent_button_plus","strength_button_minus","sense_button_minus","eloquent_button_minus","points_display"]
        res = {i:gr.update(visible=False) for i in visible_components}
        res["chat_display"] = self.get_historical_chat()
        res["send_btn"]=gr.update(interactive=True)
        res["prompt_button"]=gr.update(interactive=True)
        res["start_btn"]=gr.update(value="重新开始",interactive=True)
        res["message_input"]=gr.update(value=None,interactive=True)
        res["save_button"]=gr.update(interactive=True)
        res["bgm"]=gr.update(value=None,interactive=False,autoplay=False)
        self.is_changed_bgm = True
        
        res.update(self.update_npc_list())
        return self.update_outputs(**res)


if __name__=="__main__":
    bgms = {'bgm1':44100,
    'bgm2':32000,
    'bgm3':32000,
    'bgm4':44100,
    'bgm6':48000,
    'bgm7':44100,
    'bgm8':44100,
    'bgm9':32000,
    'bgm10':32000,
    'bgm11':44100,
    'bgm12':44100,
    'attack':44100,
    'truth':32000,
    'critical':44100}
    for k,v in bgms.items():
        bgm_numpy = np.load(k+".npy")
        value = (v,bgm_numpy)
        bgms[k] = value

    with open("profiles.json") as f:
        profiles = json.load(f)
    with open("illustrations.json") as f:
        illustrations = json.load(f)
    with open("finalCSS.css",encoding="utf-8") as f:
        css = f.read()

    config = configparser.ConfigParser()
    config.read('config.ini')
    model = config["API-Config"]["model"]
    similar_model = config["API-Config"]["similar_model"]
    key = config["API-Config"]["key"]
    client = OpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    similar_client = OpenAI(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    df_locations=pd.read_csv("df_locations.csv",dtype=df_locations_dtype)
    df_events=pd.read_csv("df_events.csv",dtype=df_events_dtype)
    df_npc=pd.read_csv("df_npc.csv",dtype=df_npc_dtype)
    df_stage_end=pd.read_csv("df_stage_end.csv",dtype=df_stage_end_dtype)
    df_prompt=pd.read_csv("df_prompt.csv",dtype=df_prompt_dtype)

    game = GameUI(df_locations,df_events,df_npc,df_stage_end,df_prompt)