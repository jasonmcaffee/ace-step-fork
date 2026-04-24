---
base_model: ACE-Step/Ace-Step1.5
library_name: peft
license: creativeml-openrail-m
tags:
- text2audio,
- music
---

<a href="https://arxiv.org/abs/2602.00744">Tech Report</a>

# New Year Themed - Female

This LoRA model was trained with its default configuration on 12 New Year-themed songs for approximately 1 hour on an A100 GPU, with a batch size of 1.
It enables style cover versions of songs with a festive New Year atmosphere, emulating the vocal style of a female singer renowned for her Chinese folk music performances.
Additionally, it can generate common Chinese folk instruments (such as the bass drum, dizi, and erhu) and support original folk-style composition.

## Training Song List
- 万事如意 (All the Best)
- 好日子 (Good Days)
- 好运来 (May Fortune Come)
- 常回家看看 (Go Home Often)
- 恭喜发财 (Wishing You a Prosperous New Year)
- 拥军花鼓 (March of the Army and the Flower Drum)
- 春节序曲 (Spring Festival Overture)
- 步步高 (Step by Step Up)
- 祝酒歌 (Toast Song)
- 越来越好 (Getting Better and Better)
- 迎宾曲 (Welcome Overture)
- 难忘今宵 (Unforgettable Tonight)


The aforementioned training data is not included in the base model’s training corpus and consists entirely of unseen new data for the base model.
This LoRA is developed for experimental purposes only, to validate the base model’s LoRA fine-tuning capabilities.
It is intended solely for research and academic exchange use, and commercial use is prohibited.
In case you believe this LoRA infringes on your legitimate rights and interests, please contact us promptly, and we will take it down immediately.

## Usage Instructions

After loading this LoRA into the ACE-Step model, only the DiT model (not the Think model) is to be used for audio synthesis.

### Recommended Caption Format & Style
```
An explosive and theatrical big band arrangement kicks off with a flurry of woodblock percussion and a dramatic orchestral hit, launching into a high-energy swing groove. A powerful female vocalist, singing in a clear and operatic style, soars over a dense mix of punchy brass stabs, a walking bassline, and a driving drum kit. The track is punctuated by virtuosic saxophone solos and fills that weave through the powerful arrangement. The overall mood is celebratory and grand, reminiscent of a classic show tune or a festive, big-band pop song with a strong, driving rhythm.
```

```
An energetic and celebratory Chinese folk-pop track driven by a powerful, galloping percussion ensemble of large drums and sharp cymbals. A piercing suona carries the main melodic hook with a bright, festive tone. A clear, powerful female vocal delivers the lyrics in a traditional, almost operatic style. The arrangement is dense and consistently high-energy, structured with instrumental introductions and interludes that showcase the suona's vibrant melody against the driving, complex percussion.
```