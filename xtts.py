from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Your long text
long_text = """
The University of Glasgow (abbreviated as Glas. in post-nominals; Scottish Gaelic: Oilthigh Ghlaschu[4]) is a public research university in Glasgow, Scotland. Founded by papal bull in 1451 [O.S. 1450],[5] it is the fourth-oldest university in the English-speaking world and one of Scotland's four ancient universities. Along with the universities of St Andrews, Aberdeen, and Edinburgh, the university was part of the Scottish Enlightenment during the 18th century. Glasgow is the largest university in Scotland by total enrolment[3] and with over 19,500 postgraduates the second-largest in the United Kingdom by postgraduate enrolment.[3]

In common with universities of the pre-modern era, Glasgow originally educated students primarily from wealthy backgrounds; however, it became a pioneer[citation needed] in British higher education in the 19th century by also providing for the needs of students from the growing urban and commercial middle class. Glasgow University served all of these students by preparing them for professions: law, medicine, civil service, teaching, and the church. It also trained smaller but growing numbers for careers in science and engineering.[6] Glasgow has the fifth-largest endowment of any university in the UK and the annual income of the institution for 2022–23 was £944.2 million of which £220.7 million was from research grants and contracts, with an expenditure of £827.4 million.[1] It is a member of Universitas 21, the Russell Group[7] and the Guild of European Research-Intensive Universities.

The university was originally located in the city's High Street; since 1870, its main campus has been at Gilmorehill in the City's West End.[8] Additionally, a number of university buildings are located elsewhere, such as the Veterinary School in Bearsden, and the Crichton Campus in Dumfries.[9]

The alumni of the University of Glasgow include some of the major figures of modern history, including James Wilson, a signatory of the United States Declaration of Independence, 3 Prime Ministers of the United Kingdom (William Lamb, Henry Campbell-Bannerman and Bonar Law), 3 Scottish First Ministers (Humza Yousaf, Nicola Sturgeon and Donald Dewar), economist Adam Smith, philosopher Francis Hutcheson, engineer James Watt, physicist Lord Kelvin, surgeon Joseph Lister along with 4 Nobel Prize laureates (in total 8 Nobel Prize winners are affiliated with the University) and numerous Olympic gold medallists, including the current chancellor, Dame Katherine Granger.
"""

# Generate speech by cloning a voice using default settings
tts.tts_to_file(text=long_text,
                file_path="output.wav",
                speaker_wav=["Recording.wav"],
                language="en",
                )
