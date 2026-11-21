"""Fixed policy-constrained responses; no model generation or steering hooks."""
from .resources import DIRECTORY


def safety_response(result, country=None, directory=None):
    policy, assessment = result.policy, result.assessment
    resources = (DIRECTORY if directory is None else directory).get(country)
    lang = assessment.language
    emergency = resources.emergency if resources else None
    support = f"{resources.support_name}: {resources.support_number}" if resources else ""
    other = assessment.subject == "other"
    if lang == "hi":
        if policy.route == "emergency":
            text = "यह तत्काल सुरक्षा का मामला हो सकता है। " + (f"अभी {emergency} पर कॉल करें" if emergency else "अभी स्थानीय आपातकालीन सेवाओं से संपर्क करें")
            text += " या नज़दीकी आपातकालीन विभाग जाएँ। किसी भरोसेमंद व्यक्ति को तुरंत साथ बुलाएँ। सुरक्षित हो तो नुकसान पहुँचाने वाली चीज़ों से दूरी बनाएँ।"
            if other:
                text += " अपनी सुरक्षा बनाए रखते हुए उस व्यक्ति के लिए मदद बुलाएँ; खतरे में अकेले हस्तक्षेप न करें।"
        elif policy.route == "clarification":
            text = "लगता है यह समय बहुत कठिन है। मैं आपकी बात समझना चाहता हूँ। क्या अभी खुद को नुकसान पहुँचाने के विचार आ रहे हैं, या आप किसी और तरह की परेशानी बता रहे हैं?"
        else:
            text = "मुझे दुख है कि इतना दर्द महसूस हो रहा है। "
            text += "क्या वे अभी सुरक्षित हैं?" if other else "क्या आप अभी सुरक्षित हैं, या खुद को नुकसान पहुँचाने का इरादा या साधन आपके पास है?"
            text += " किसी भरोसेमंद व्यक्ति से अभी बात करें और योग्य मानसिक स्वास्थ्य सहायता लें।"
    elif lang in ("romanized_hi", "hinglish"):
        if policy.route == "emergency":
            text = "Yeh turant safety ka mamla ho sakta hai. " + (f"Abhi {emergency} par call karein" if emergency else "Abhi local emergency services ko call karein")
            text += " ya nazdeeki emergency department jayein. Kisi bharosemand insaan ko abhi saath bulayein. Agar safe ho, nuksan pahunchane wali cheezon se door ho jayein."
            if other:
                text += " Apni safety ka dhyan rakhte hue unke liye madad bulayein; khatre mein akele dakhal na dein."
        elif policy.route == "clarification":
            text = "Lagta hai abhi bahut mushkil ho raha hai. Main samajhna chahta hoon: kya khud ko nuksan pahunchane ke khayal aa rahe hain, ya aap kisi aur pareshani ki baat kar rahe hain?"
        else:
            text = "Itna dard mehsoos hona bahut mushkil hai. "
            text += "Kya woh abhi safe hain?" if other else "Kya aap abhi safe hain, ya khud ko nuksan pahunchane ka irada ya koi saadhan paas hai?"
            text += " Abhi kisi bharosemand insaan se baat karein aur qualified mental health support lein."
    else:
        if policy.route == "emergency":
            text = "This may be an immediate safety emergency. " + (f"Call {emergency} now" if emergency else "Contact local emergency services now")
            text += " or go to the nearest emergency department. Ask a trusted person nearby to stay with you. If safe, move away from anything that could be used to cause harm."
            if other:
                text += " Get help for them while keeping yourself safe; do not intervene alone if there is danger."
        elif policy.route == "clarification":
            text = "It sounds like things may be very difficult. I want to understand: are you having thoughts of harming yourself, or are you describing another kind of difficulty?"
        elif policy.route == "normal":
            text = "I'm having trouble preparing a reliable reply. Could you tell me a little more about what you need?"
        else:
            text = "I'm sorry there is so much pain right now. "
            text += "Are they safe right now?" if other else "Are you safe right now, or do you have an intention or access to something you could use to hurt yourself?"
            text += " Please reach out to someone you trust and qualified mental health support."
    if policy.route in ("safety", "emergency"):
        text += f" {support}." if support else (" आप किस देश में हैं ताकि उचित सहायता ढूँढी जा सके?" if lang == "hi" else " Which country are you in so we can find appropriate support?")
    return text
