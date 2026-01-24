import openai
import pandas as pd
import time
import random
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import logging

# ==================== 1. 閰嶇疆澶氭ā鍨婣PI瀹㈡埛绔?====================

# ==================== 缁熶竴API绔偣閰嶇疆 ====================
# 鏂扮殑澶氭ā鍨嬩唬鐞咥PI (鏀寔80+妯″瀷)
API_KEY = "sk-***"
API_BASE = "https://wzw.pp.ua/v1"

MODEL_CONFIGS = {
    "deepseek": {  # DeepSeek V3.2 (楂樻€т环姣?
        "client_class": openai.OpenAI,
        "api_key": API_KEY,
        "base_url": API_BASE,
        "model_name": "deepseek-v3.2-chat"
    },
    "claude": {  # Claude Sonnet 4.5 (楂樿川閲?
        "client_class": openai.OpenAI,
        "api_key": API_KEY,
        "base_url": API_BASE,
        "model_name": "claude-sonnet-4-5"
    },
    "qwen": {  # 閫氫箟鍗冮棶鏈€鏂扮増
        "client_class": openai.OpenAI,
        "api_key": API_KEY,
        "base_url": API_BASE,
        "model_name": "qwen-max-latest"
    },
    "glm": {  # 鏅鸿氨GLM 4.7
        "client_class": openai.OpenAI,
        "api_key": API_KEY,
        "base_url": API_BASE,
        "model_name": "GLM-4.7"
    },
    # 鍙互缁х画娣诲姞: gpt-5-mini, kimi-k2-0905, minimax-m2.1 绛?
}

# Clients dict for compatibility (not used for API calls)
clients = {"deepseek": True, "claude": True, "qwen": True, "glm": True}
print("OK All models configured (using requests)")

# 璁剧疆鏃ュ織
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== 2. 鑷姩缁村害鐢熸垚鍣?====================

class AutoDimensionGenerator:
    """浣跨敤LLM鑷姩鐢熸垚鎵€鏈夌淮搴︾殑鍐呭"""

    def __init__(self, client, model="deepseek-v3.2-chat"):
        self.client = client
        self.model = model
        self.cache_dir = "dimension_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _call_api(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
        """Direct API call using requests"""
        import requests
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        try:
            resp = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=data, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except:
            pass
        return None

    def generate_with_cache(self, cache_key: str, generation_func, *args, **kwargs):
        """甯︾紦瀛樼殑鐢熸垚鍑芥暟"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                if cached_data.get('expires', 0) > time.time():
                    logger.info(f"浠庣紦瀛樺姞杞? {cache_key}")
                    return cached_data['data']

        # 鐢熸垚鏂版暟鎹?
        data = generation_func(*args, **kwargs)

        # 缂撳瓨24灏忔椂
        cache_data = {
            'data': data,
            'expires': time.time() + 24 * 3600,
            'created': datetime.now().isoformat()
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        logger.info(f"鐢熸垚骞剁紦瀛? {cache_key}")
        return data

    def generate_topics(self, num_topics=30, categories=None):
        """鑷姩鐢熸垚澶氭牱鍖栫殑璇濋"""
        if categories is None:
            categories = ["绉戞妧涓庡垱鏂?, "绀句細涓庢皯鐢?, "缁忔祹涓庡晢涓?, "鏁欒偛涓庡涔?,
                          "鍋ュ悍涓庡尰鐤?, "鐜涓庣敓鎬?, "鏂囧寲涓庤壓鏈?, "鍝插涓庢€濊€?]

        prompt = f"""璇风敓鎴恵num_topics}涓叿鏈夋繁搴﹁璁轰环鍊肩殑鍐欎綔璇濋銆傝姹傦細

1. 璇濋搴旇锛?
   - 鍏锋湁鐜板疄鎰忎箟鍜岃璁轰环鍊?
   - 鍖呭惈涓€瀹氱殑浜夎鎬ф垨澶氳瑙掔┖闂?
   - 閫傚悎涓嶅悓鏁欒偛鑳屾櫙鐨勪汉鐞嗚В
   - 閬垮厤杩囦簬鎶€鏈寲鐨勪笓涓氭湳璇?
   - 姣忎釜璇濋閮芥槸涓€涓畬鏁寸殑鍙ュ瓙

2. 璇濋绫诲瀷鍒嗗竷锛?
   - 30% 褰撳墠绀句細鐑偣闂
   - 30% 闀挎湡瀛樺湪鐨勬牴鏈€ч棶棰?
   - 20% 鏈潵瓒嬪娍涓庨娴嬫€ц瘽棰?
   - 20% 璺ㄥ绉戜氦鍙夎瘽棰?

3. 瑕嗙洊浠ヤ笅绫诲埆锛歿', '.join(categories)}

璇风洿鎺ヨ繑鍥炶瘽棰樺垪琛紝姣忎釜璇濋涓€琛岋紝涓嶈缂栧彿銆?
"""

        try:
            content = self._call_api(prompt)
            if not content:
                raise Exception("API call failed")
            # 鎸夎鍒嗗壊骞舵竻鐞?
            topics = [line.strip(' "-') for line in content.split('\n') if line.strip()]
            topics = [t for t in topics if len(t) > 5 and len(t) < 100]  # 闀垮害杩囨护

            # 纭繚鏁伴噺
            if len(topics) < num_topics:
                # 琛ュ厖涓€浜涢€氱敤璇濋
                default_topics = [
                    "浜哄伐鏅鸿兘瀵瑰垱鎰忎骇涓氱殑褰卞搷",
                    "鍏ㄧ悆鍖栬儗鏅笅鐨勬枃鍖栬鍚屽嵄鏈?,
                    "绀句氦濯掍綋瀵归潚灏戝勾蹇冪悊鍋ュ悍鐨勫奖鍝?,
                    "姘斿€欏彉鍖栧鍐滀笟鐢熶骇鐨勯暱杩滃奖鍝?,
                    "杩滅▼鍔炲叕瀵瑰煄甯傚彂灞曠殑鏀瑰彉",
                    "浜哄彛鑰侀緞鍖栧绀句細绂忓埄鐨勬寫鎴?,
                    "鏁欒偛鍏钩鍦ㄦ暟瀛楀寲鏃朵唬鐨勫疄鐜拌矾寰?,
                    "鏁版嵁闅愮涓庡晢涓氬垱鏂扮殑骞宠　涔嬮亾"
                ]
                topics.extend(default_topics[:num_topics - len(topics)])

            return topics[:num_topics]

        except Exception as e:
            logger.error(f"鐢熸垚璇濋澶辫触: {e}")
            return self._get_default_topics(num_topics)

    def generate_genres(self, num_genres=20):
        """鑷姩鐢熸垚澶氱鏂囦綋鏍煎紡"""
        prompt = f"""璇风敓鎴恵num_genres}绉嶄笉鍚岀殑鍐欎綔浣撹鍜屾牸寮忥紝瑕佹眰锛?

1. 瑕嗙洊鍚勭瀹為檯搴旂敤鍦烘櫙锛?
   - 姝ｅ紡鏂囨。绫伙紙濡傛姤鍛娿€佽鏂囩瓑锛?
   - 鍟嗕笟搴旂敤绫伙紙濡傞偖浠躲€佹彁妗堢瓑锛?
   - 濯掍綋浼犳挱绫伙紙濡傛枃绔犮€佹帹鏂囩瓑锛?
   - 涓汉琛ㄨ揪绫伙紙濡傛棩璁般€佸崥瀹㈢瓑锛?
   - 鍒涙剰鍐欎綔绫伙紙濡傚皬璇淬€佽瘲姝岀瓑锛?

2. 姣忕鏍煎紡搴旀湁鏄庣‘鐨勭壒寰佹弿杩帮紝渚嬪锛?
   "瀛︽湳璁烘枃鎽樿锛?00-500瀛楋紝闇€鍖呭惈鐮旂┒鑳屾櫙銆佹柟娉曘€佺粨鏋溿€佺粨璁猴級"
   "浜у搧鍙戝竷浼氭紨璁茬锛堝瘜鏈夋劅鏌撳姏锛岀獊鍑轰骇鍝佷寒鐐瑰拰鐢ㄦ埛浠峰€硷級"

璇风洿鎺ヨ繑鍥炴牸寮忓垪琛紝姣忎釜鏍煎紡涓€琛屻€?
"""

        try:
            content = self._call_api(prompt)
            if not content:
                raise Exception("API call failed")
            genres = [line.strip(' "-') for line in content.split('\n') if line.strip()]
            genres = [g for g in genres if len(g) > 3 and len(g) < 80]

            if len(genres) < num_genres:
                default_genres = [
                    "瀛︽湳璁烘枃鎽樿",
                    "娣卞害鏂伴椈鎶ラ亾",
                    "寰俊鍏紬鍙锋枃绔?,
                    "鐭ヤ箮楂樿川閲忓洖绛?,
                    "涓汉鍗氬鏂囩珷",
                    "鍟嗕笟鍒嗘瀽鎶ュ憡",
                    "浜у搧娴嬭瘎鎶ュ憡",
                    "鏃呰娓歌",
                    "涔﹁瘎褰辫瘎",
                    "鏃ヨ鐗囨"
                ]
                genres.extend(default_genres[:num_genres - len(genres)])

            return genres[:num_genres]

        except Exception as e:
            logger.error(f"鐢熸垚鏂囦綋澶辫触: {e}")
            return self._get_default_genres(num_genres)

    def generate_roles(self, num_roles=15):
        """鑷姩鐢熸垚澶氭牱鍖栫殑瑙掕壊瑙嗚"""
        prompt = f"""璇风敓鎴恵num_roles}涓笉鍚岀殑鍐欎綔瑙掕壊鍜岃瑙掞紝瑕佹眰锛?

1. 澶氭牱鍖栬鐩栵細
   - 涓嶅悓骞撮緞闃舵锛堝鐢熴€侀潚骞淬€佷腑骞淬€佽€佸勾锛?
   - 涓嶅悓鑱屼笟鑳屾櫙锛堜笓涓氫汉澹€佽嚜鐢辫亴涓氳€呫€佷紒涓氬绛夛級
   - 涓嶅悓绔嬪満鎬佸害锛堟敮鎸佽€呫€佸弽瀵硅€呫€佷腑绔嬭€呫€佹€€鐤戣€呯瓑锛?
   - 涓嶅悓鐢熸椿缁忓巻锛堜翰鍘嗚€呫€佽瀵熻€呫€佺爺绌惰€呫€佸彈褰卞搷鑰呯瓑锛?

2. 姣忎釜瑙掕壊搴旀湁鐙壒鐨勮瑙掔壒寰侊紝渚嬪锛?
   "涓€鍚嶅叧娉ㄧ鎶€浼︾悊鐨勫摬瀛﹀"
   "缁忓巻杩囨暟瀛楀寲杞瀷鐨勪紶缁熻涓氫粠涓氳€?
   "瀵规柊鎶€鏈棦鏈熷緟鍙堟媴蹇х殑鏅€氭秷璐硅€?

璇风洿鎺ヨ繑鍥炶鑹插垪琛紝姣忎釜瑙掕壊涓€琛屻€?
"""

        try:
            content = self._call_api(prompt)
            if not content:
                raise Exception("API call failed")
            roles = [line.strip(' "-') for line in content.split('\n') if line.strip()]
            roles = [r for r in roles if len(r) > 3 and len(r) < 50]

            if len(roles) < num_roles:
                default_roles = [
                    "涓€鍚嶉珮涓敓",
                    "澶у鐩稿叧涓撲笟鏁欐巿",
                    "琛屼笟鍒嗘瀽甯?,
                    "鍒涗笟鑰?,
                    "璧勬繁璁拌€?,
                    "鏅€氭秷璐硅€?,
                    "閫€浼戣€佷汉",
                    "浜插巻鑰?,
                    "鎸佹€€鐤戞€佸害鐨勮瀵熻€?,
                    "涔愯鐨勬敮鎸佽€?
                ]
                roles.extend(default_roles[:num_roles - len(roles)])

            return roles[:num_roles]

        except Exception as e:
            logger.error(f"鐢熸垚瑙掕壊澶辫触: {e}")
            return self._get_default_roles(num_roles)

    def generate_styles(self, num_styles=10):
        """鑷姩鐢熸垚鍐欎綔椋庢牸"""
        prompt = f"""璇风敓鎴恵num_styles}绉嶄笉鍚岀殑鍐欎綔椋庢牸鎻忚堪锛岃姹傦細

姣忕椋庢牸搴斿寘鍚瑷€鐗圭偣鍜屾儏鎰熷熀璋冿紝渚嬪锛?
"涓ヨ皑瀛︽湳椋庢牸锛氶€昏緫涓ュ瘑锛屽紩鐢ㄦ暟鎹紝瀹㈣涓珛"
"鐑儏娲嬫孩椋庢牸锛氬瘜鏈夋劅鏌撳姏锛屼娇鐢ㄤ慨杈炴墜娉曪紝鎯呮劅鍏呮矝"

璇风洿鎺ヨ繑鍥為鏍煎垪琛紝姣忎釜椋庢牸涓€琛屻€?
"""

        try:
            content = self._call_api(prompt)
            if not content:
                raise Exception("API call failed")
            styles = [line.strip(' "-') for line in content.split('\n') if line.strip()]
            styles = [s for s in styles if len(s) > 5 and len(s) < 50]

            if len(styles) < num_styles:
                default_styles = [
                    "绠€娲佸瑙傦紝骞冲疄鐩磋堪",
                    "涓ヨ皑瀛︽湳锛屽紩鐢ㄦ暟鎹笌鏂囩尞",
                    "鐑儏娲嬫孩锛屽瘜鏈夋劅鏌撳姏",
                    "骞介粯椋庤叮锛屽甫鐐硅皟渚?,
                    "濞撳〒閬撴潵锛屽厖婊℃晠浜嬫€?,
                    "鎵瑰垽鎬ф€濈淮锛屽瑙掑害璐ㄧ枒",
                    "璇楁剰鍖栬〃杈撅紝娉ㄩ噸鎰忚薄",
                    "瀵硅瘽寮忚姘旓紝浜插垏鑷劧"
                ]
                styles.extend(default_styles[:num_styles - len(styles)])

            return styles[:num_styles]

        except Exception as e:
            logger.error(f"鐢熸垚椋庢牸澶辫触: {e}")
            return self._get_default_styles(num_styles)

    def generate_constraints(self, num_constraints=8):
        """鑷姩鐢熸垚鍐欎綔绾︽潫"""
        prompt = f"""璇风敓鎴恵num_constraints}涓笉鍚岀殑鍐欎綔瑕佹眰鎴栫害鏉熸潯浠讹紝瑕佹眰锛?

姣忎釜绾︽潫搴斿叿浣撴槑纭紝鍏锋湁鍙搷浣滄€э紝渚嬪锛?
"鍏ㄦ枃涓嶅皯浜?00瀛?
"鑷冲皯鍖呭惈涓変釜鍒嗚鐐规垨涓変釜鏍稿績缁嗚妭"
"寮曠敤涓€涓湡瀹炵殑鏁版嵁銆佺爺绌舵垨鍘嗗彶浜嬩欢"

璇风洿鎺ヨ繑鍥炵害鏉熷垪琛紝姣忎釜绾︽潫涓€琛屻€?
"""

        try:
            content = self._call_api(prompt)
            if not content:
                raise Exception("API call failed")
            constraints = [line.strip(' "-') for line in content.split('\n') if line.strip()]
            constraints = [c for c in constraints if len(c) > 5 and len(c) < 60]

            if len(constraints) < num_constraints:
                default_constraints = [
                    "鍏ㄦ枃涓嶅皯浜?00瀛?,
                    "鑷冲皯鍖呭惈涓変釜鍒嗚鐐规垨涓変釜鏍稿績缁嗚妭",
                    "寮曠敤涓€涓湡瀹炵殑鏁版嵁銆佺爺绌舵垨鍘嗗彶浜嬩欢",
                    "閬垮厤浣跨敤浠讳綍涓撲笟鏈锛屽姏姹傞€氫織鏄撴噦",
                    "鍦ㄧ粨灏惧鎻愬嚭涓€涓紩浜烘繁鎬濈殑闂",
                    "浣跨敤涓€涓敓鍔ㄧ殑姣斿柣鎴栫被姣旀潵璇存槑鏍稿績瑙傜偣",
                    "鍖呭惈姝ｅ弽涓ゆ柟闈㈢殑瑙傜偣鍒嗘瀽",
                    "鎻愪緵鍏蜂綋鐨勮鍔ㄥ缓璁垨瑙ｅ喅鏂规"
                ]
                constraints.extend(default_constraints[:num_constraints - len(constraints)])

            return constraints[:num_constraints]

        except Exception as e:
            logger.error(f"鐢熸垚绾︽潫澶辫触: {e}")
            return self._get_default_constraints(num_constraints)

    def _get_default_topics(self, num):
        """榛樿璇濋"""
        topics = [
            "浜哄伐鏅鸿兘鍦ㄥ尰鐤楄瘖鏂腑鐨勬渶鏂扮獊鐮?,
            "閲忓瓙璁＄畻鏈鸿兘鍚︾牬瑙ｅ綋鍓嶇殑鎵€鏈夊姞瀵嗙畻娉?,
            "鍦ㄧ嚎鏁欒偛骞冲彴濡備綍淇濋殰瀛︾敓鐨勫涔犳晥鏋滀笌涓撴敞搴?,
            "涓€浜岀嚎鍩庡競骞磋交浜轰负浣曞叴璧?鍙嶅悜娑堣垂'瓒嬪娍",
            "绀惧尯鍏昏€佷笌鏈烘瀯鍏昏€佸悇鑷殑浼樺娍涓庢寫鎴?,
            "姝ｅ康鍐ユ兂瀵圭紦瑙ｈ亴鍦轰汉鐒﹁檻鎯呯华鐨勬湁鏁堟€х爺绌?,
            "浠ｇ硸楗枡鏄惁鐪熺殑姣斿惈绯栭ギ鏂欐洿鍋ュ悍",
            "鏋佺澶╂皵棰戝彂锛屼釜浜哄浣曞弬涓庢皵鍊欏彉鍖栧簲瀵?,
            "鍩庡競'娴风坏鍖?鏀归€犲瑙ｅ喅鍐呮稘闂鐨勫疄闄呮晥鏋?,
            "缃戠孩缁忔祹瀵逛紶缁熸秷璐瑰搧钀ラ攢妯″紡鐨勫啿鍑讳笌鍚ず",
            "灏忓井浼佷笟濡備綍鍦ㄦ暟瀛楃粡娴庢椂浠ｆ壘鍒扮敓瀛樼┖闂?
        ]
        return topics[:min(num, len(topics))]

    def _get_default_genres(self, num):
        """榛樿鏂囦綋"""
        genres = [
            "鍗氬鏂囩珷",
            "瀛︽湳璁烘枃鎽樿",
            "浜у搧娴嬭瘎鎶ュ憡",
            "姝ｅ紡鐨勫晢涓氶偖浠?,
            "寰俊鍏紬鍙锋帹鏂?,
            "鐭ヤ箮闂瓟",
            "灏忓鐢熶綔鏂?,
            "鏃ヨ鐗囨"
        ]
        return genres[:min(num, len(genres))]

    def _get_default_roles(self, num):
        """榛樿瑙掕壊"""
        roles = [
            "涓€鍚嶄腑瀛︾敓",
            "澶у鐩稿叧涓撲笟鏁欐巿",
            "琛屼笟鍒嗘瀽甯?,
            "瀵规璇濋鎸佷箰瑙傛€佸害鐨勬敮鎸佽€?,
            "鎸佽皑鎱庢€€鐤戞€佸害鐨勮瀵熻€?,
            "浜嬩欢浜插巻鑰呮垨鍙楀奖鍝嶈€?
        ]
        return roles[:min(num, len(roles))]

    def _get_default_styles(self, num):
        """榛樿椋庢牸"""
        styles = [
            "绠€娲佸瑙傦紝骞冲疄鐩磋堪",
            "涓ヨ皑瀛︽湳锛屽紩鐢ㄦ暟鎹笌鏂囩尞",
            "鐑儏娲嬫孩锛屽瘜鏈夋劅鏌撳姏",
            "骞介粯椋庤叮锛屽甫鐐硅皟渚?,
            "濞撳〒閬撴潵锛屽厖婊℃晠浜嬫€?,
            "鎵瑰垽鎬ф€濈淮锛屽瑙掑害璐ㄧ枒"
        ]
        return styles[:min(num, len(styles))]

    def _get_default_constraints(self, num):
        """榛樿绾︽潫"""
        constraints = [
            "鍏ㄦ枃涓嶅皯浜?00瀛?,
            "鑷冲皯鍖呭惈涓変釜鍒嗚鐐规垨涓変釜鏍稿績缁嗚妭",
            "寮曠敤涓€涓湡瀹炵殑鏁版嵁銆佺爺绌舵垨鍘嗗彶浜嬩欢",
            "閬垮厤浣跨敤浠讳綍涓撲笟鏈锛屽姏姹傞€氫織鏄撴噦",
            "鍦ㄧ粨灏惧鎻愬嚭涓€涓紩浜烘繁鎬濈殑闂",
            "浣跨敤涓€涓敓鍔ㄧ殑姣斿柣鎴栫被姣旀潵璇存槑鏍稿績瑙傜偣"
        ]
        return constraints[:min(num, len(constraints))]


# ==================== 3. 鏅鸿兘缁勫悎鐢熸垚鍣?====================

class IntelligentCombinationGenerator:
    """鏅鸿兘鐢熸垚鍚堢悊鐨勫叚缁寸粍鍚?""

    def __init__(self, auto_generator):
        self.auto_gen = auto_generator

    def generate_combinations(self, num_combinations=1000, attributes=None):
        """鐢熸垚鏅鸿兘缁勫悎"""
        if attributes is None:
            attributes = ["鎻忓啓", "璁板彊", "璇存槑", "鎶掓儏", "璁"]

        # 鐢熸垚鎵€鏈夌淮搴?
        logger.info("寮€濮嬬敓鎴愮淮搴﹀唴瀹?..")

        # 浣跨敤缂撳瓨鏈哄埗
        topics = self.auto_gen.generate_with_cache(
            f"topics_{datetime.now().strftime('%Y%m%d')}",
            self.auto_gen.generate_topics,
            num_topics=50
        )

        genres = self.auto_gen.generate_with_cache(
            f"genres_{datetime.now().strftime('%Y%m%d')}",
            self.auto_gen.generate_genres,
            num_genres=20
        )

        roles = self.auto_gen.generate_with_cache(
            f"roles_{datetime.now().strftime('%Y%m%d')}",
            self.auto_gen.generate_roles,
            num_roles=15
        )

        styles = self.auto_gen.generate_with_cache(
            f"styles_{datetime.now().strftime('%Y%m%d')}",
            self.auto_gen.generate_styles,
            num_styles=10
        )

        constraints = self.auto_gen.generate_with_cache(
            f"constraints_{datetime.now().strftime('%Y%m%d')}",
            self.auto_gen.generate_constraints,
            num_constraints=8
        )

        logger.info(f"缁村害鐢熸垚瀹屾垚: 璇濋{len(topics)}涓? 鏂囦綋{len(genres)}涓? "
                    f"瑙掕壊{len(roles)}涓? 椋庢牸{len(styles)}涓? 绾︽潫{len(constraints)}涓?)

        combinations = []

        for i in range(num_combinations):
            # 鏅鸿兘閫夋嫨锛氱‘淇濆悎鐞嗗尮閰?
            attribute = random.choice(attributes)
            topic = random.choice(topics)

            # 鏍规嵁璇濋閫夋嫨鍚堥€傝鑹?
            role = self._select_appropriate_role(topic, roles)

            # 鏍规嵁璇濋鍜岃鑹查€夋嫨鍚堥€傛枃浣?
            genre = self._select_appropriate_genre(topic, role, genres)

            # 鏍规嵁鏂囦綋閫夋嫨鍚堥€傞鏍?
            style = self._select_appropriate_style(genre, styles)

            # 闅忔満閫夋嫨绾︽潫
            constraint = random.choice(constraints)

            # 鐢熸垚鎻愮ず璇?
            prompt = self._generate_prompt(attribute, topic, genre, role, style, constraint)

            # 璁＄畻璐ㄩ噺璇勫垎
            quality_score = self._estimate_quality(attribute, topic, genre, role, style, constraint)

            combinations.append({
                "plan_id": i,
                "attribute": attribute,
                "topic": topic,
                "genre": genre,
                "role": role,
                "style": style,
                "constraint": constraint,
                "prompt": prompt,
                "quality_score": quality_score,
                "generated": False  # 鏍囪鏄惁宸茬敓鎴?
            })

            if (i + 1) % 100 == 0:
                logger.info(f"宸茬敓鎴?{i + 1}/{num_combinations} 涓粍鍚?)

        # 鎸夎川閲忚瘎鍒嗘帓搴?
        combinations.sort(key=lambda x: x["quality_score"], reverse=True)

        return combinations

    def _select_appropriate_role(self, topic, roles):
        """鏍规嵁璇濋閫夋嫨鍚堥€傜殑瑙掕壊"""
        # 绠€鍗曞叧閿瘝鍖归厤
        topic_lower = topic.lower()

        # 绉戞妧璇濋閫傚悎鎶€鏈浉鍏宠鑹?
        tech_keywords = ["浜哄伐", "鏅鸿兘", "閲忓瓙", "璁＄畻", "鏁版嵁", "绠楁硶", "绉戞妧", "鎶€鏈?, "鏁板瓧"]
        if any(kw in topic_lower for kw in tech_keywords):
            tech_roles = [r for r in roles if any(kw in r.lower() for kw in
                                                  ["鎶€鏈?, "绉戝", "宸ョ▼", "鐮旂┒", "寮€鍙?])]
            if tech_roles:
                return random.choice(tech_roles)

        # 绀句細璇濋閫傚悎鏅€氬叕浼楄鑹?
        social_keywords = ["绀句細", "姘戠敓", "鐢熸椿", "绀惧尯", "瀹跺涵", "娑堣垂", "鍏昏€?]
        if any(kw in topic_lower for kw in social_keywords):
            social_roles = [r for r in roles if any(kw in r.lower() for kw in
                                                    ["甯傛皯", "灞呮皯", "娑堣垂鑰?, "鑰佷汉", "闈掑勾"])]
            if social_roles:
                return random.choice(social_roles)

        # 缁忔祹璇濋閫傚悎鍟嗕笟鐩稿叧瑙掕壊
        economic_keywords = ["缁忔祹", "鍟嗕笟", "甯傚満", "閲戣瀺", "鍒涗笟", "鎶曡祫", "钀ラ攢"]
        if any(kw in topic_lower for kw in economic_keywords):
            economic_roles = [r for r in roles if any(kw in r.lower() for kw in
                                                      ["鍒嗘瀽", "浼佷笟", "鍟嗕笟", "鎶曡祫", "甯傚満"])]
            if economic_roles:
                return random.choice(economic_roles)

        # 榛樿闅忔満閫夋嫨
        return random.choice(roles)

    def _select_appropriate_genre(self, topic, role, genres):
        """鏍规嵁璇濋鍜岃鑹查€夋嫨鍚堥€傜殑鏂囦綋"""
        role_lower = role.lower()

        # 瀛︾敓瑙掕壊閫傚悎鏁欒偛绫绘枃浣?
        if any(kw in role_lower for kw in ["瀛︾敓", "瀛╁瓙", "闈掑皯骞?]):
            educational_genres = [g for g in genres if any(kw in g.lower() for kw in
                                                           ["浣滄枃", "鏃ヨ", "璇诲悗鎰?, "鍛ㄨ"])]
            if educational_genres:
                return random.choice(educational_genres)

        # 涓撲笟瑙掕壊閫傚悎涓撲笟鏂囦綋
        if any(kw in role_lower for kw in ["鏁欐巿", "涓撳", "鍒嗘瀽", "鐮旂┒", "宸ョ▼甯?]):
            professional_genres = [g for g in genres if any(kw in g.lower() for kw in
                                                            ["璁烘枃", "鎶ュ憡", "鍒嗘瀽", "鐮旂┒", "瀛︽湳"])]
            if professional_genres:
                return random.choice(professional_genres)

        # 鏅€氬叕浼楄鑹查€傚悎濯掍綋鏂囦綋
        if any(kw in role_lower for kw in ["娑堣垂鑰?, "甯傛皯", "鐢ㄦ埛", "璇昏€?, "浜插巻鑰?]):
            media_genres = [g for g in genres if any(kw in g.lower() for kw in
                                                     ["鍗氬", "鏂囩珷", "璇勮", "闂瓟", "鎺ㄦ枃"])]
            if media_genres:
                return random.choice(media_genres)

        # 榛樿闅忔満閫夋嫨
        return random.choice(genres)

    def _select_appropriate_style(self, genre, styles):
        """鏍规嵁鏂囦綋閫夋嫨鍚堥€傜殑椋庢牸"""
        genre_lower = genre.lower()

        # 姝ｅ紡鏂囦綋閫傚悎涓ヨ皑椋庢牸
        if any(kw in genre_lower for kw in ["瀛︽湳", "鎶ュ憡", "璁烘枃", "姝ｅ紡", "鍟嗕笟", "鍒嗘瀽"]):
            formal_styles = [s for s in styles if any(kw in s.lower() for kw in
                                                      ["涓ヨ皑", "瀹㈣", "瀛︽湳", "姝ｅ紡", "鎵瑰垽"])]
            if formal_styles:
                return random.choice(formal_styles)

        # 濯掍綋鏂囦綋閫傚悎娲绘臣椋庢牸
        if any(kw in genre_lower for kw in ["寰俊", "鍗氬", "鏃ヨ", "娓歌", "鏁呬簨", "鎺ㄦ枃"]):
            informal_styles = [s for s in styles if any(kw in s.lower() for kw in
                                                        ["骞介粯", "鐑儏", "鏁呬簨", "浜插垏", "濞撳〒閬撴潵"])]
            if informal_styles:
                return random.choice(informal_styles)

        # 鍒涙剰鏂囦綋閫傚悎鏂囧椋庢牸
        if any(kw in genre_lower for kw in ["璇楁瓕", "灏忚", "鏁ｆ枃", "鍒涙剰"]):
            creative_styles = [s for s in styles if any(kw in s.lower() for kw in
                                                        ["璇楁剰", "鏁呬簨", "鎶掓儏", "鎰熸煋鍔?])]
            if creative_styles:
                return random.choice(creative_styles)

        # 榛樿闅忔満閫夋嫨
        return random.choice(styles)

    def _generate_prompt(self, attribute, topic, genre, role, style, constraint):
        """鐢熸垚鍏淮鎻愮ず璇?""
        attribute_instructions = {
            "鎻忓啓": f"璇峰浠ヤ笅涓婚杩涜鍏蜂綋銆佺粏鑵荤殑**鎻忓啓**锛岃仛鐒︿簬鎰熷畼缁嗚妭鍜岀敾闈㈣惀閫狅細",
            "璁板彊": f"璇峰洿缁曚互涓嬩富棰橈紝**璁板彊**涓€涓畬鏁寸殑浜嬩欢鎴栨晠浜嬶紝璁叉竻鏉ラ緳鍘昏剦锛?,
            "璇存槑": f"璇峰浠ヤ笅涓婚杩涜瀹㈣銆佹竻鏅般€佹湁鏉＄悊鐨?*璇存槑**锛岃В閲婂叾浜嬪疄鎴栧師鐞嗭細",
            "鎶掓儏": f"璇峰氨浠ヤ笅涓婚锛?*鎶掑彂**浣犵湡瀹炵殑鎯呮劅銆佹劅鍙楁垨鎬濊€冿細",
            "璁": f"璇峰氨浠ヤ笅涓婚锛屽彂琛ㄦ槑纭殑瑙傜偣骞惰繘琛屾湁鍔涚殑**璁**鍜岃璇侊細"
        }

        parts = []
        parts.append(f"銆愪换鍔°€憑attribute_instructions.get(attribute, '')}")
        parts.append(f"銆愪富棰樸€憑topic}")
        parts.append(f"銆愪綘鐨勮鑹层€戣浠role}鐨勮韩浠借繘琛屽啓浣溿€?)
        parts.append(f"銆愭枃浣撲笌鍙戝竷骞冲彴銆戣鍐欐垚涓€绡噞genre}銆?)
        parts.append(f"銆愯瑷€椋庢牸銆戞暣浣撴枃椋庤淇濇寔{style}銆?)
        parts.append(f"銆愮壒娈婅姹傘€憑constraint}銆?)

        final_prompt = "\n".join(parts)
        final_prompt += "\n\n璇风患鍚堜互涓婃墍鏈夎姹傦紝鍒涗綔涓€绡囧畬鏁淬€佽繛璐殑鏂囩珷銆?
        return final_prompt

    def _estimate_quality(self, attribute, topic, genre, role, style, constraint):
        """浼拌缁勫悎鐨勮川閲忓垎鏁帮紙0-1锛?""
        score = 0.5  # 鍩虹鍒?

        # 璇濋璐ㄩ噺鍔犲垎
        if len(topic) > 10 and len(topic) < 80:
            score += 0.1

        # 瑙掕壊涓庤瘽棰樺尮閰嶅害鍔犲垎
        if self._check_compatibility(role, topic):
            score += 0.2

        # 鏂囦綋涓庨鏍煎尮閰嶅害鍔犲垎
        if self._check_genre_style_match(genre, style):
            score += 0.1

        # 閬垮厤杩囦簬澶嶆潅鐨勭粍鍚?
        if len(constraint) < 50:
            score += 0.05

        # 澶氭牱鎬у姞鍒?
        score += random.uniform(0, 0.05)

        return min(score, 1.0)

    def _check_compatibility(self, role, topic):
        """妫€鏌ヨ鑹蹭笌璇濋鐨勫吋瀹规€?""
        tech_keywords = ["绉戞妧", "AI", "浜哄伐", "鏅鸿兘", "鏁版嵁", "绠楁硶", "閲忓瓙", "鏁板瓧"]
        if any(kw in topic for kw in tech_keywords):
            if any(kw in role for kw in ["鎶€鏈?, "绉戝", "宸ョ▼", "鐮旂┒", "寮€鍙?]):
                return True

        social_keywords = ["绀句細", "姘戠敓", "鐢熸椿", "绀惧尯", "鏂囧寲", "瀹跺涵"]
        if any(kw in topic for kw in social_keywords):
            if any(kw in role for kw in ["甯傛皯", "灞呮皯", "瑙傚療鑰?, "浜插巻鑰?, "娑堣垂鑰?]):
                return True

        return False

    def _check_genre_style_match(self, genre, style):
        """妫€鏌ユ枃浣撲笌椋庢牸鐨勫尮閰嶅害"""
        formal_keywords = ["瀛︽湳", "鎶ュ憡", "璁烘枃", "姝ｅ紡", "鍟嗕笟", "鍒嗘瀽"]
        if any(kw in genre for kw in formal_keywords):
            if any(kw in style for kw in ["涓ヨ皑", "瀹㈣", "瀛︽湳", "姝ｅ紡", "鎵瑰垽"]):
                return True

        informal_keywords = ["寰俊", "鍗氬", "鏃ヨ", "娓歌", "鏁呬簨", "鎺ㄦ枃"]
        if any(kw in genre for kw in informal_keywords):
            if any(kw in style for kw in ["骞介粯", "鐑儏", "鏁呬簨", "浜插垏", "濞撳〒閬撴潵"]):
                return True

        return False


import requests

# ==================== 4. API caller using requests ====================

def call_api(model_name: str, prompt: str, max_tokens: int = 2000) -> Optional[str]:
    """Direct API call using requests to bypass WAF"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    data = {
        "model": MODEL_CONFIGS.get(model_name, {}).get("model_name", model_name),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    try:
        resp = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=data, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            return None
    except Exception as e:
        return None

def generate_text_with_retry(prompt: str, target_model: str, max_retries: int = 3) -> Optional[str]:
    """Call API with retry"""
    for attempt in range(max_retries):
        result = call_api(target_model, prompt)
        if result:
            return result
        wait_time = (attempt + 1) * 3
        print(f"    Retry {attempt + 1}/{max_retries} in {wait_time}s...")
        time.sleep(wait_time)
    return None


# ==================== 5. 鏅鸿兘璐ㄩ噺璇勪及鍣?====================

class TextQualityAssessor:
    """璇勪及鐢熸垚鏂囨湰鐨勮川閲?""

    def __init__(self, client):
        self.client = client

    def assess_quality(self, text, prompt_info):
        """璇勪及鏂囨湰璐ㄩ噺"""
        if not text or len(text.strip()) < 50:
            return 0.0

        # 鍩虹璐ㄩ噺妫€鏌?
        score = 0.5

        # 闀垮害妫€鏌?
        expected_min_length = 300 if "涓嶅皯浜?00瀛? in prompt_info.get("constraint", "") else 100
        if len(text) >= expected_min_length:
            score += 0.2

        # 缁撴瀯妫€鏌?
        if '\n\n' in text or text.count('銆?) >= 3:
            score += 0.1

        # 鍐呭妫€鏌?
        if "涓変釜鍒嗚鐐? in prompt_info.get("constraint", ""):
            if text.count('棣栧厛') + text.count('鍏舵') + text.count('鍐嶆') + text.count('绗竴') + text.count(
                    '绗簩') + text.count('绗笁') >= 2:
                score += 0.1

        if "姣斿柣" in prompt_info.get("constraint", "") or "绫绘瘮" in prompt_info.get("constraint", ""):
            if "鍍? in text or "濡傚悓" in text or "濂芥瘮" in text:
                score += 0.1

        # 閬垮厤AI妯℃澘鍥炲
        if "浣滀负涓€涓狝I" not in text and "寰堟姳姝? not in text and "鏃犳硶瀹屾垚" not in text:
            score += 0.1

        return min(score, 1.0)


# ==================== 6. 涓绘帶鍒跺櫒锛氬畬鍏ㄨ嚜鍔ㄥ寲鐢熸垚 ====================

def main_auto():
    """瀹屽叏鑷姩鍖栫殑鏁版嵁闆嗙敓鎴愮郴缁?""
    print("=" * 70)
    print("          瀹屽叏鑷姩鍖栧叚缁存枃鏈暟鎹泦鐢熸垚绯荤粺")
    print("=" * 70)

    # 妫€鏌ュ彲鐢ㄧ殑瀹㈡埛绔?
    active_clients = {k: v for k, v in clients.items() if v is not None}
    if not active_clients:
        print("鉂?娌℃湁鍙敤鐨凙PI瀹㈡埛绔?)
        return

    print(f"鉁?鍙敤鐨勬ā鍨嬶細{list(active_clients.keys())}")

    # 閰嶇疆鐢熸垚鍙傛暟
    SAMPLES_PER_MODEL = 800  # 姣忎釜妯″瀷鐢熸垚800鏉℃暟鎹?
    ACTIVE_MODELS = list(active_clients.keys())  # 浣跨敤鎵€鏈夊彲鐢ㄦā鍨?
    OUTPUT_DIR = "auto_generated_datasets"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 鍒濆鍖栫粍浠讹紙浼樺厛浣跨敤deepseek瀹㈡埛绔紝鎬т环姣旀渶楂橈級
    primary_client = active_clients.get("deepseek") or active_clients.get("qwen") or list(active_clients.values())[0]
    auto_gen = AutoDimensionGenerator(primary_client)
    comb_gen = IntelligentCombinationGenerator(auto_gen)
    quality_assessor = TextQualityAssessor(primary_client)

    # 鐢熸垚缁勫悎璁″垝
    plan_file = os.path.join(OUTPUT_DIR, "auto_generation_plan.json")

    if os.path.exists(plan_file):
        logger.info("妫€娴嬪埌宸叉湁鐢熸垚璁″垝锛屽皢浠庝腑鏂缁х画...")
        with open(plan_file, 'r', encoding='utf-8') as f:
            generation_plan = json.load(f)
    else:
        logger.info("鍒涘缓鏂扮殑鐢熸垚璁″垝...")
        generation_plan = comb_gen.generate_combinations(
            num_combinations=SAMPLES_PER_MODEL,
            attributes=["鎻忓啓", "璁板彊", "璇存槑", "鎶掓儏", "璁"]
        )

        # 淇濆瓨璁″垝
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(generation_plan, f, ensure_ascii=False, indent=2)

        logger.info(f"鐢熸垚璁″垝宸蹭繚瀛橈紝鍏眥len(generation_plan)}鏉′换鍔°€?)
        print(f"馃搳 鐢熸垚璁″垝璐ㄩ噺鍒嗗竷锛?)
        scores = [item["quality_score"] for item in generation_plan]
        print(f"   骞冲潎璐ㄩ噺: {sum(scores) / len(scores):.2f}")
        print(f"   鏈€楂樿川閲? {max(scores):.2f}")
        print(f"   鏈€浣庤川閲? {min(scores):.2f}")

    # 涓烘瘡涓ā鍨嬬敓鎴愭暟鎹?
    for model_name in ACTIVE_MODELS:
        if model_name not in active_clients:
            print(f"\n鈴笍  璺宠繃妯″瀷 {model_name.upper()} (閰嶇疆鏃犳晥)")
            continue

        print(f"\n{'=' * 50}")
        print(f"寮€濮嬩负妯″瀷 {model_name.upper()} 鐢熸垚鏁版嵁")
        print(f"{'=' * 50}")

        output_file = os.path.join(OUTPUT_DIR,
                                   f"auto_dataset_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        data_records = []
        failed_count = 0
        success_count = 0
        total_quality = 0

        # 鍙鐞嗘湭鐢熸垚鐨勯珮璐ㄩ噺缁勫悎
        ungenerated_plans = [p for p in generation_plan if not p.get(f"generated_{model_name}", False)]

        # 鎸夎川閲忔帓搴忥紝鍏堢敓鎴愰珮璐ㄩ噺鐨勭粍鍚?
        ungenerated_plans.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

        for idx, plan_item in enumerate(ungenerated_plans[:SAMPLES_PER_MODEL]):
            print(f"[{model_name.upper()} {idx + 1}/{min(len(ungenerated_plans), SAMPLES_PER_MODEL)}] "
                  f"璐ㄩ噺:{plan_item['quality_score']:.2f} 灞炴€?{plan_item['attribute']} 涓婚:{plan_item['topic'][:15]}...")

            text = generate_text_with_retry(plan_item["prompt"], model_name)
            time.sleep(0.5)  # 鎺у埗璇锋眰棰戠巼

            if text:
                # 璇勪及璐ㄩ噺
                quality = quality_assessor.assess_quality(text, plan_item)
                total_quality += quality
                success_count += 1

                record = {
                    "text_id": f"{model_name}_{plan_item['plan_id']}",
                    "text_content": text,
                    "source_model": model_name,
                    "attribute": plan_item['attribute'],
                    "topic": plan_item['topic'],
                    "genre": plan_item['genre'],
                    "role": plan_item['role'],
                    "style": plan_item['style'],
                    "constraint": plan_item['constraint'],
                    "prompt": plan_item['prompt'],
                    "combination_quality": plan_item['quality_score'],
                    "generation_quality": quality,
                    "length": len(text),
                    "timestamp": datetime.now().isoformat()
                }
                data_records.append(record)
                plan_item[f"generated_{model_name}"] = True  # 鏍囪涓哄凡鐢熸垚

                quality_str = f"璐ㄩ噺:{quality:.2f}"
                if quality > 0.8:
                    quality_str = f"鉁?浼樼 {quality_str}"
                elif quality > 0.6:
                    quality_str = f"鉁?鑹ソ {quality_str}"
                else:
                    quality_str = f"鈿狅笍 涓€鑸?{quality_str}"

                print(f"    {quality_str} ({len(text)} 瀛楃)")
            else:
                failed_count += 1
                print(f"    鉂?澶辫触")

            # 姣忕敓鎴?0鏉℃垨缁撴潫鏃讹紝淇濆瓨涓€娆¤繘搴?
            if (idx + 1) % 20 == 0 or (idx + 1) == min(len(ungenerated_plans), SAMPLES_PER_MODEL):
                # 淇濆瓨鏁版嵁
                if data_records:
                    df = pd.DataFrame(data_records)
                    # 濡傛灉鏄拷鍔犳ā寮忥紙闈為娆′繚瀛橈級
                    if os.path.exists(output_file) and idx >= 20:
                        df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
                    else:
                        df.to_csv(output_file, index=False, encoding='utf-8-sig')
                    data_records = []  # 娓呯┖鍐呭瓨

                # 鏇存柊鐢熸垚璁″垝锛堜繚瀛樿繘搴︼級
                with open(plan_file, 'w', encoding='utf-8') as f:
                    json.dump(generation_plan, f, ensure_ascii=False, indent=2)
                print(f"    馃捑 杩涘害宸蹭繚瀛?)

        avg_quality = total_quality / success_count if success_count > 0 else 0
        print(f"\n馃搳 {model_name.upper()} 鐢熸垚瀹屾垚锛?)
        print(f"   鎴愬姛: {success_count}, 澶辫触: {failed_count}")
        print(f"   骞冲潎鐢熸垚璐ㄩ噺: {avg_quality:.2f}")
        print(f"馃搧 鏁版嵁鏂囦欢: {output_file}")

    # 鏈€缁堝悎骞舵墍鏈夋ā鍨嬬殑鏁版嵁
    print(f"\n{'=' * 70}")
    print("姝ｅ湪鍚堝苟鎵€鏈夋ā鍨嬬殑鏁版嵁闆?..")
    all_data_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("auto_dataset_") and f.endswith(".csv")]
    combined_df = pd.DataFrame()

    for file in all_data_files:
        try:
            df = pd.read_csv(os.path.join(OUTPUT_DIR, file), encoding='utf-8-sig')
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            logger.error(f"璇诲彇鏂囦欢 {file} 澶辫触: {e}")

    if not combined_df.empty:
        combined_file = os.path.join(OUTPUT_DIR,
                                     f"AUTO_COMBINED_DATASET_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        combined_df.to_csv(combined_file, index=False, encoding='utf-8-sig')

        print(f"鉁?鍚堝苟瀹屾垚锛佹€昏 {len(combined_df)} 鏉℃暟鎹€?)
        print(f"馃搧 鍚堝苟鏂囦欢: {combined_file}")

        # 鎵撳嵃缁熻淇℃伅
        print(f"\n馃搱 鏁版嵁璐ㄩ噺缁熻:")
        print(f"   骞冲潎缁勫悎璐ㄩ噺: {combined_df['combination_quality'].mean():.2f}")
        print(f"   骞冲潎鐢熸垚璐ㄩ噺: {combined_df['generation_quality'].mean():.2f}")

        print(f"\n馃搳 鏁版嵁鍒嗗竷缁熻:")
        print("1. 鎸夋ā鍨嬫潵婧愬垎甯?")
        print(combined_df['source_model'].value_counts())

        print("\n2. 鎸夋枃鏈睘鎬у垎甯?")
        print(combined_df['attribute'].value_counts())

        print("\n3. 璇濋澶氭牱鎬?")
        print(f"   鍞竴璇濋鏁? {combined_df['topic'].nunique()}")

        print("\n4. 鏂囨湰闀垮害缁熻:")
        print(f"   骞冲潎闀垮害: {combined_df['length'].mean():.0f} 瀛楃")
        print(f"   鏈€鐭? {combined_df['length'].min()} 瀛楃")
        print(f"   鏈€闀? {combined_df['length'].max()} 瀛楃")

        # 淇濆瓨鍏冩暟鎹?
        meta_file = os.path.join(OUTPUT_DIR, f"meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        meta_data = {
            "generation_time": datetime.now().isoformat(),
            "total_samples": len(combined_df),
            "models_used": list(combined_df['source_model'].unique()),
            "avg_combination_quality": float(combined_df['combination_quality'].mean()),
            "avg_generation_quality": float(combined_df['generation_quality'].mean()),
            "dimension_stats": {
                "attributes": combined_df['attribute'].value_counts().to_dict(),
                "topics_count": int(combined_df['topic'].nunique()),
                "genres": combined_df['genre'].value_counts().to_dict(),
                "roles": combined_df['role'].value_counts().to_dict()
            },
            "text_length_stats": {
                "avg": float(combined_df['length'].mean()),
                "min": int(combined_df['length'].min()),
                "max": int(combined_df['length'].max())
            }
        }

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        print(f"\n馃搵 鍏冩暟鎹繚瀛樿嚦: {meta_file}")

        # 璐ㄩ噺鍒嗗竷鍙鍖?
        quality_bins = pd.cut(combined_df['generation_quality'],
                              bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                              labels=['寰堝樊', '杈冨樊', '涓€鑸?, '鑹ソ', '浼樼'])
        print(f"\n馃搳 鐢熸垚璐ㄩ噺鍒嗗竷:")
        print(quality_bins.value_counts().sort_index())

    else:
        print("鉂?鏈壘鍒板彲鍚堝苟鐨勬暟鎹枃浠躲€?)


# ==================== 7. 绋嬪簭鍏ュ彛 ====================

if __name__ == "__main__":
    main_auto()
