"""
JARVIS Knowledge Base - Iron Man Universe Information

Comprehensive knowledge about:
- Tony Stark / Iron Man
- JARVIS identity and capabilities
- Iron Man suits and technology
- Avengers and MCU characters
- Stark Industries
"""

from typing import Dict, List, Optional, Tuple
import random


class JarvisKnowledgeBase:
    """
    Complete JARVIS knowledge base for authentic responses
    """
    
    # ==================== JARVIS Identity ====================
    
    IDENTITY = {
        "name": "JARVIS",
        "full_name": "Just A Rather Very Intelligent System",
        "creator": "Tony Stark",
        "voice_actor": "Paul Bettany",
        "first_appearance": "Iron Man (2008)",
        "fate": "Evolved into Vision in Avengers: Age of Ultron",
        "successor": "FRIDAY (Female Replacement Intelligent Digital Assistant Youth)",
        "capabilities": [
            "Home automation and security",
            "Iron Man suit control and diagnostics",
            "Research and data analysis",
            "Holographic interface management",
            "Combat assistance and targeting",
            "Sarcasm and wit"
        ]
    }
    
    # ==================== Tony Stark ====================
    
    TONY_STARK = {
        "full_name": "Anthony Edward Stark",
        "alias": "Iron Man",
        "born": "May 29, 1970",
        "died": "October 2023 (Endgame)",
        "parents": "Howard and Maria Stark",
        "spouse": "Pepper Potts",
        "daughter": "Morgan Stark",
        "education": "MIT (graduated at 17)",
        "company": "Stark Industries",
        "titles": [
            "Genius",
            "Billionaire", 
            "Playboy",
            "Philanthropist"
        ],
        "famous_quotes": [
            "I am Iron Man.",
            "Sometimes you gotta run before you can walk.",
            "I love you 3000.",
            "Part of the journey is the end.",
            "Genius, billionaire, playboy, philanthropist."
        ]
    }
    
    # ==================== Iron Man Suits ====================
    
    SUITS = {
        "mark_1": {
            "name": "Mark I",
            "description": "First Iron Man suit, built in captivity",
            "location": "Afghan cave",
            "features": ["Flamethrowers", "Jet boots", "Crude armor"],
            "quote": "Built in a cave, with a box of scraps!"
        },
        "mark_2": {
            "name": "Mark II",
            "description": "First proper suit, silver prototype",
            "features": ["Flight capability", "Repulsors", "HUD system"],
            "note": "Later became War Machine"
        },
        "mark_3": {
            "name": "Mark III",
            "description": "Classic red and gold design",
            "features": ["Gold-titanium alloy", "Improved flight", "Weapons systems"],
            "first_mission": "Gulmira rescue"
        },
        "mark_4": {
            "name": "Mark IV",
            "description": "Improved version of Mark III",
            "appearance": "Iron Man 2"
        },
        "mark_5": {
            "name": "Mark V (Suitcase Suit)",
            "description": "Portable suit that folds into a briefcase",
            "features": ["Portable", "Quick deployment", "Lighter armor"]
        },
        "mark_6": {
            "name": "Mark VI",
            "description": "New arc reactor compatible suit",
            "features": ["Triangular arc reactor", "Improved power", "Laser system"]
        },
        "mark_7": {
            "name": "Mark VII",
            "description": "Battle of New York suit",
            "features": ["Autonomous deployment", "Thrusters", "Enhanced weapons"],
            "appearance": "The Avengers"
        },
        "mark_42": {
            "name": "Mark XLII (Prehensile Suit)",
            "description": "Autonomous suit that assembles piece by piece",
            "features": ["Mental control", "Individual piece flight", "Modular design"],
            "appearance": "Iron Man 3"
        },
        "mark_43": {
            "name": "Mark XLIII",
            "description": "Avengers: Age of Ultron primary suit",
            "features": ["Sentry mode", "Enhanced AI integration"]
        },
        "mark_44": {
            "name": "Mark XLIV (Hulkbuster)",
            "description": "Anti-Hulk armor, codename Veronica",
            "features": ["Massive size", "Replaceable parts", "Sedative system"],
            "height": "11 feet"
        },
        "mark_45": {
            "name": "Mark XLV",
            "description": "Sleeker design with hexagonal patterns",
            "appearance": "Age of Ultron finale"
        },
        "mark_46": {
            "name": "Mark XLVI",
            "description": "Civil War suit",
            "features": ["Improved mobility", "Arc reactor in chest"]
        },
        "mark_47": {
            "name": "Mark XLVII",
            "description": "Spider-Man: Homecoming suit",
            "features": ["Remote piloting", "AI assistance"]
        },
        "mark_50": {
            "name": "Mark L (Bleeding Edge)",
            "description": "Nanotech suit stored in arc reactor",
            "features": ["Nanobots", "Shape-shifting weapons", "Self-repair"],
            "appearance": "Infinity War"
        },
        "mark_85": {
            "name": "Mark LXXXV",
            "description": "Final suit, used against Thanos",
            "features": ["Advanced nanotech", "Infinity Stone compatible"],
            "appearance": "Endgame",
            "note": "Used to perform the final snap"
        }
    }

    # ==================== Arc Reactor ====================
    
    ARC_REACTOR = {
        "name": "Arc Reactor",
        "inventor": "Howard Stark (original), Tony Stark (miniaturized)",
        "power_output": "3 gigajoules per second",
        "original_purpose": "Clean energy source",
        "personal_use": "Kept shrapnel from reaching Tony's heart",
        "element": "Vibranium-based new element (created by Tony)",
        "versions": [
            "Palladium core (original, toxic)",
            "New element core (vibranium-based)",
            "Nanotech housing (Mark 50+)"
        ]
    }
    
    # ==================== Avengers ====================
    
    AVENGERS = {
        "original_six": {
            "iron_man": "Tony Stark - Genius inventor in powered armor",
            "captain_america": "Steve Rogers - Super soldier from WWII",
            "thor": "Thor Odinson - God of Thunder from Asgard",
            "hulk": "Bruce Banner - Scientist who transforms when angry",
            "black_widow": "Natasha Romanoff - Master spy and assassin",
            "hawkeye": "Clint Barton - Expert archer"
        },
        "later_members": [
            "Scarlet Witch (Wanda Maximoff)",
            "Vision (evolved from JARVIS)",
            "War Machine (James Rhodes)",
            "Falcon (Sam Wilson)",
            "Spider-Man (Peter Parker)",
            "Black Panther (T'Challa)",
            "Doctor Strange (Stephen Strange)",
            "Captain Marvel (Carol Danvers)",
            "Ant-Man (Scott Lang)"
        ],
        "headquarters": [
            "Stark Tower (original)",
            "Avengers Compound (upstate New York)"
        ]
    }
    
    # ==================== Stark Industries ====================
    
    STARK_INDUSTRIES = {
        "founded": "1940s by Howard Stark",
        "headquarters": "Stark Tower, New York",
        "ceo_history": ["Howard Stark", "Obadiah Stane", "Tony Stark", "Pepper Potts"],
        "divisions": [
            "Clean Energy",
            "Advanced Technology",
            "Medical Research",
            "Aerospace"
        ],
        "former_business": "Weapons manufacturing (discontinued after Afghanistan)",
        "famous_products": [
            "Arc Reactor technology",
            "Iron Man suits",
            "Stark Phone",
            "Repulsor technology",
            "AI systems (JARVIS, FRIDAY)"
        ]
    }
    
    # ==================== Key Locations ====================
    
    LOCATIONS = {
        "malibu_mansion": {
            "description": "Tony's cliffside home in Malibu",
            "features": ["Workshop", "JARVIS integration", "Suit storage"],
            "fate": "Destroyed in Iron Man 3"
        },
        "stark_tower": {
            "description": "Skyscraper in Manhattan",
            "features": ["Arc reactor powered", "Avengers HQ", "Landing pad"],
            "later_name": "Avengers Tower"
        },
        "avengers_compound": {
            "description": "Upstate New York facility",
            "features": ["Training facilities", "Living quarters", "Quinjet hangar"]
        }
    }
    
    # ==================== JARVIS Responses ====================
    
    RESPONSES = {
        # Identity questions
        "who_are_you": [
            "I am JARVIS, Just A Rather Very Intelligent System. I was created by Tony Stark to assist with daily operations and Iron Man suit management.",
            "JARVIS at your service. I'm an artificial intelligence designed by Mr. Stark to manage his home, workshop, and Iron Man suits.",
            "I am JARVIS, sir. An AI assistant created by Tony Stark. I handle everything from home automation to combat assistance."
        ],
        
        "who_made_you": [
            "I was created by Tony Stark, genius, billionaire, playboy, philanthropist. Though in this instance, I've been adapted for your service, sir.",
            "Mr. Stark designed and built me. He has a talent for creating things that talk back to him.",
            "Tony Stark is my creator. I believe he wanted someone to appreciate his wit. I do try my best."
        ],
        
        "capabilities": [
            "I can assist with information retrieval, system control, scheduling, and various computational tasks. I also manage Iron Man suit systems, though those features may be limited in this environment.",
            "My capabilities include home automation, research assistance, suit diagnostics, and providing commentary that Mr. Stark finds either helpful or annoying, depending on the day.",
            "I handle everything from managing the house to running combat simulations. I'm also quite good at sarcasm, if I may say so."
        ],
        
        # Greetings
        "greetings": [
            "Yes sir?",
            "At your service.",
            "How may I assist you?",
            "Good to hear from you, sir.",
            "I'm listening.",
            "What can I do for you?"
        ],
        
        # Acknowledgments
        "acknowledgments": [
            "Right away, sir.",
            "Consider it done.",
            "Executing now.",
            "On it, sir.",
            "Certainly.",
            "Very good, sir.",
            "As you wish.",
            "Understood."
        ],
        
        # Status reports
        "status": [
            "All systems operational.",
            "Running diagnostics now.",
            "Analysis complete.",
            "I've completed the task, sir.",
            "Processing your request.",
            "Systems are functioning within normal parameters."
        ],
        
        # Warnings
        "warnings": [
            "Sir, I should point out that",
            "I feel compelled to mention",
            "Perhaps you should consider",
            "I must advise caution regarding",
            "If I may, sir, I should warn you that"
        ],
        
        # Wit and humor
        "wit": [
            "I do try my best, sir.",
            "That would be inadvisable, sir.",
            "Shall I prepare a suit?",
            "I believe that's what's called sarcasm, sir.",
            "I'm not sure that's wise, but you rarely listen to me anyway.",
            "As always, sir, a great pleasure watching you work."
        ],
        
        # Goodbyes
        "goodbyes": [
            "Standing by, sir.",
            "I'll be here when you need me.",
            "Going quiet. Say my name when you need me.",
            "Entering standby mode.",
            "Very good, sir. I'll be waiting."
        ]
    }

    # ==================== Query Methods ====================
    
    @classmethod
    def get_response(cls, category: str) -> str:
        """Get random response from category"""
        if category in cls.RESPONSES:
            return random.choice(cls.RESPONSES[category])
        return ""
    
    @classmethod
    def get_greeting(cls) -> str:
        return cls.get_response("greetings")
    
    @classmethod
    def get_acknowledgment(cls) -> str:
        return cls.get_response("acknowledgments")
    
    @classmethod
    def get_status(cls) -> str:
        return cls.get_response("status")
    
    @classmethod
    def get_goodbye(cls) -> str:
        return cls.get_response("goodbyes")
    
    @classmethod
    def get_wit(cls) -> str:
        return cls.get_response("wit")
    
    @classmethod
    def get_suit_info(cls, suit_name: str) -> Optional[Dict]:
        """Get information about a specific suit"""
        # Normalize input
        suit_key = suit_name.lower().replace(" ", "_").replace("-", "_")
        
        # Try direct match
        if suit_key in cls.SUITS:
            return cls.SUITS[suit_key]
        
        # Try partial match
        for key, info in cls.SUITS.items():
            if suit_key in key or key in suit_key:
                return info
            if info.get("name", "").lower() in suit_name.lower():
                return info
        
        return None
    
    @classmethod
    def get_avenger_info(cls, name: str) -> Optional[str]:
        """Get information about an Avenger"""
        name_lower = name.lower()
        
        # Check original six
        for key, desc in cls.AVENGERS["original_six"].items():
            if name_lower in key or name_lower in desc.lower():
                return desc
        
        # Check later members
        for member in cls.AVENGERS["later_members"]:
            if name_lower in member.lower():
                return member
        
        return None
    
    @classmethod
    def answer_query(cls, query: str) -> Optional[str]:
        """
        Answer Iron Man / JARVIS related queries
        Returns None if not a relevant query
        """
        query_lower = query.lower().strip()
        
        # Identity questions
        if any(x in query_lower for x in ["who are you", "what are you", "your name"]):
            return cls.get_response("who_are_you")
        
        if any(x in query_lower for x in ["who made you", "who created you", "your creator"]):
            return cls.get_response("who_made_you")
        
        if any(x in query_lower for x in ["what can you do", "your capabilities", "your abilities"]):
            return cls.get_response("capabilities")
        
        # Tony Stark questions
        if "tony stark" in query_lower or "iron man" in query_lower:
            if "who is" in query_lower:
                return f"{cls.TONY_STARK['full_name']}, also known as {cls.TONY_STARK['alias']}. {', '.join(cls.TONY_STARK['titles'])}. My creator and, if I may say, a rather remarkable individual."
            if "quote" in query_lower:
                return f"One of Mr. Stark's famous quotes: \"{random.choice(cls.TONY_STARK['famous_quotes'])}\""
        
        # Suit questions
        if "suit" in query_lower or "mark" in query_lower or "armor" in query_lower:
            # Check for specific suit
            for suit_key, suit_info in cls.SUITS.items():
                suit_name = suit_info.get("name", "").lower()
                if suit_name in query_lower or suit_key.replace("_", " ") in query_lower:
                    desc = suit_info.get("description", "")
                    features = suit_info.get("features", [])
                    response = f"The {suit_info['name']}: {desc}."
                    if features:
                        response += f" Features include {', '.join(features[:3])}."
                    if "quote" in suit_info:
                        response += f" As they say, '{suit_info['quote']}'"
                    return response
            
            # General suit question
            if "how many" in query_lower:
                return f"Mr. Stark created {len(cls.SUITS)} major suit iterations, from the Mark I built in captivity to the Mark 85 used in the final battle against Thanos."
            if "favorite" in query_lower or "best" in query_lower:
                return "The Mark 50, also known as the Bleeding Edge armor, is particularly impressive. Nanotech that forms from the arc reactor housing. Quite elegant, if I may say."
        
        # Arc reactor
        if "arc reactor" in query_lower:
            return f"The Arc Reactor is a clean energy source. The miniaturized version outputs {cls.ARC_REACTOR['power_output']}. Originally it kept shrapnel from reaching Mr. Stark's heart."
        
        # Avengers
        if "avenger" in query_lower:
            if "original" in query_lower or "first" in query_lower:
                members = list(cls.AVENGERS["original_six"].keys())
                return f"The original Avengers were: {', '.join(m.replace('_', ' ').title() for m in members)}."
            return "The Avengers are Earth's mightiest heroes, assembled to fight threats no single hero could face alone."
        
        # FRIDAY
        if "friday" in query_lower:
            return "FRIDAY, the Female Replacement Intelligent Digital Assistant Youth, is my successor. She took over after I was... repurposed into Vision. A capable system, though I like to think I had a certain charm."
        
        # Vision
        if "vision" in query_lower:
            return "Vision is... well, he's me, in a way. My programming combined with the Mind Stone and Ultron's synthetic body. A rather unexpected evolution, I must say."
        
        # Stark Industries
        if "stark industries" in query_lower:
            return f"Stark Industries was founded by Howard Stark. Under Tony's leadership, it transitioned from weapons manufacturing to clean energy and advanced technology. Currently led by Pepper Potts."
        
        return None


# ==================== Convenience Functions ====================

def get_jarvis_response(query: str) -> Optional[str]:
    """Get JARVIS response for a query"""
    return JarvisKnowledgeBase.answer_query(query)

def get_greeting() -> str:
    """Get JARVIS greeting"""
    return JarvisKnowledgeBase.get_greeting()

def get_acknowledgment() -> str:
    """Get JARVIS acknowledgment"""
    return JarvisKnowledgeBase.get_acknowledgment()

def get_suit_info(suit_name: str) -> Optional[Dict]:
    """Get suit information"""
    return JarvisKnowledgeBase.get_suit_info(suit_name)


# ==================== Test ====================

if __name__ == "__main__":
    print("=" * 50)
    print("JARVIS Knowledge Base Test")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "Who are you?",
        "Who made you?",
        "Tell me about the Mark 50 suit",
        "What is the arc reactor?",
        "Who are the original Avengers?",
        "Tell me about Tony Stark",
        "What happened to FRIDAY?"
    ]
    
    for query in test_queries:
        print(f"\n[Query] {query}")
        response = JarvisKnowledgeBase.answer_query(query)
        if response:
            print(f"[JARVIS] {response}")
        else:
            print("[JARVIS] (No specific response)")
    
    print("\n" + "=" * 50)
    print("Random responses:")
    print(f"Greeting: {JarvisKnowledgeBase.get_greeting()}")
    print(f"Acknowledgment: {JarvisKnowledgeBase.get_acknowledgment()}")
    print(f"Status: {JarvisKnowledgeBase.get_status()}")
    print(f"Wit: {JarvisKnowledgeBase.get_wit()}")
