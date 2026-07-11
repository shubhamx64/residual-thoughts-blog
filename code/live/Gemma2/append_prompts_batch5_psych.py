import csv
import os

prompts = [
    {
        "category": "Psychology",
        "topic": "Bipolar Disorder",
        "description": "Overview of Bipolar Disorder, its types, symptoms, and treatments.",
        "content": """Bipolar disorder, previously known as manic depression, is a mental health condition that causes extreme mood swings that include emotional highs (mania or hypomania) and lows (depression). When you become depressed, you may feel sad or hopeless and lose interest or pleasure in most activities. When your mood shifts to mania or hypomania (less extreme than mania), you may feel euphoric, full of energy or unusually irritable. These mood swings can affect sleep, energy, activity, judgment, behavior and the ability to think clearly.

Episodes of mood swings may occur rarely or multiple times a year. While most people will experience some emotional symptoms between episodes, some may not notice any. Although bipolar disorder is a lifelong condition, you can manage your mood swings and other symptoms by following a treatment plan. In most cases, bipolar disorder is treated with medications and psychological counseling (psychotherapy).

There are several types of bipolar and related disorders. They may include mania or hypomania and depression. Symptoms can cause unpredictable changes in mood and behavior, resulting in significant distress and difficulty in life.
*   **Bipolar I Disorder:** You've had at least one manic episode that may be preceded or followed by hypomanic or major depressive episodes. In some cases, mania may trigger a break from reality (psychosis).
*   **Bipolar II Disorder:** You've had at least one major depressive episode and at least one hypomanic episode, but you've never had a manic episode.
*   **Cyclothymic Disorder:** You've had at least two years — or one year in children and teenagers — of many periods of hypomania symptoms and periods of depressive symptoms (though less severe than major depression).

**Manic and Hypomanic Episodes:**
Mania and hypomania are two distinct types of episodes, but they have the same symptoms. Mania is more severe than hypomania and causes more noticeable problems at work, school and social activities, as well as relationship difficulties. Mania may also trigger a break from reality (psychosis) and require hospitalization. Both a manic and a hypomanic episode include three or more of these symptoms:
*   Abnormally upbeat, jumpy or wired.
*   Increased activity, energy or agitation.
*   Exaggerated sense of well-being and self-confidence (euphoria).
*   Decreased need for sleep.
*   Unusual talkativeness.
*   Racing thoughts.
*   Distractibility.

**Major Depressive Episode:**
A major depressive episode includes symptoms that are severe enough to cause noticeable difficulty in day-to-day activities, such as work, school, social activities or relationships. An episode includes five or more of these symptoms:
*   Depressed mood, such as feeling sad, empty, hopeless or tearful.
*   Marked loss of interest or feeling no pleasure in all — or almost all — activities.
*   Significant weight loss when not dieting, weight gain, or decrease or increase in appetite.
*   Either insomnia or sleeping too much.
*   Either restlessness or slowed behavior.
*   Fatigue or loss of energy.
*   Feelings of worthlessness or excessive or inappropriate guilt.
*   Decreased ability to think or concentrate, or indecisiveness.
*   Thinking about, planning or attempting suicide.

Scientists are studying the possible causes of bipolar disorder. Most agree that there is no single cause. Rather, it is likely that many factors act together to produce the illness or increase risk. Brain structure and functioning: Some studies show how the brains of people with bipolar disorder may differ from the brains of healthy people or people with other mental disorders. Genetics: Some research suggests that people with certain genes are more likely to develop bipolar disorder than others. But genes are not the only risk factor for bipolar disorder. Studies of identical twins have shown that even if one twin develops bipolar disorder, the other twin does not always develop the disorder.

Treatment helps many people, even those with the most severe forms of bipolar disorder. An effective treatment plan usually includes a combination of medication and psychotherapy, also called "talk therapy." Bipolar disorder is a lifelong illness. Episodes of mania and depression typically come back over time. Between episodes, many people with bipolar disorder are free of mood changes, but some people may have lingering symptoms. Long-term, continuous treatment helps to control these symptoms."""
    },
    {
        "category": "Psychology",
        "topic": "Borderline Personality Disorder (BPD)",
        "description": "Characteristics, diagnosis, and emotional regulation challenges in BPD.",
        "content": """Borderline Personality Disorder (BPD) is a mental health condition marked by a pattern of ongoing instability in moods, behavior, self-image, and functioning. These experiences often result in impulsive actions and unstable relationships. A person with BPD may experience intense episodes of anger, depression, and anxiety that generally last from a few hours to days.

People with BPD may experience mood swings and display uncertainty about how they see themselves and their role in the world. As a result, their interests and values can change quickly. They also tend to view things in extremes, such as all good or all bad. Their opinions of other people can also change quickly. An individual who is seen as a friend one day may be considered an enemy or traitor the next. These shifting feelings can lead to intense and unstable relationships.

Other signs or symptoms may include:
*   An intense fear of abandonment, even going to extreme measures to avoid real or imagined separation or rejection.
*   A pattern of unstable intense relationships, such as idealizing someone one moment and then suddenly believing the person does not care enough or is cruel.
*   Rapid changes in self-identity and self-image that include shifting goals and values, and seeing yourself as bad or as if you don't exist at all.
*   Periods of stress-related paranoia and loss of contact with reality, lasting from a few minutes to a few hours.
*   Impulsive and risky behavior, such as gambling, reckless driving, unsafe sex, spending sprees, binge eating or drug abuse, or sabotaging success by suddenly quitting a good job or ending a positive relationship.
*   Suicidal threats or behavior or self-injury, often in response to fear of separation or rejection.
*   Wide mood swings lasting from a few hours to a few days, which can include intense happiness, irritability, shame or anxiety.
*   Ongoing feelings of emptiness.
*   Inappropriate, intense anger, such as frequent losing of temper, being sarcastic or bitter, or having physical fights.

The causes of BPD are not fully understood. However, scientists agree that it is the result of a combination of factors, including:
*   **Genetics:** While no specific gene has been shown to cause BPD, studies suggest that the disorder has strong hereditary links. BPD is about five times more likely to occur if a person has a close family member (first-degree biological relative) with the disorder.
*   **Environmental Factors:** People who experience traumatic life events (e.g., physical or sexual abuse during childhood, neglect and separation from parents) have an increased risk of developing BPD.
*   **Brain Function:** The way the brain works is often different in people with BPD, suggesting that there is a neurological basis for some of the symptoms. Specifically, the portions of the brain that control emotions and decision-making/judgment may not communicate optimally with one another.

Historically, BPD has been viewed as difficult to treat. However, with newer, evidence-based treatment, many people with BPD experience fewer and less severe symptoms, and can function better in their daily lives.
**Psychotherapy** is the first-line treatment for BPD. One specialized form of therapy developed specifically for BPD is **Dialectical Behavior Therapy (DBT)**. DBT focuses on teaching skills to control intense emotions, reduce self-destructive behaviors, and improve relationships. It is based on the concept of dialectics—balancing opposites (acceptance of oneself and the need for change).
**Cognitive Behavioral Therapy (CBT)** can help people with BPD identify and change core beliefs and behaviors that underlie inaccurate perceptions of themselves and others and problems interacting with others.

Medication acts as a support to psychotherapy. While there is no single medication specifically for BPD, mood stabilizers, antidepressants, and antipsychotics may be prescribed to help manage specific symptoms like mood swings, depression, or disorganized thinking."""
    },
    {
        "category": "Psychology",
        "topic": "Schizophrenia",
        "description": "Symptoms, causes, and mechanisms of Schizophrenia.",
        "content": """Schizophrenia is a serious mental disorder in which people interpret reality abnormally. Schizophrenia may result in some combination of hallucinations, delusions, and extremely disordered thinking and behavior that impairs daily functioning, and can be disabling. People with schizophrenia require lifelong treatment. Early treatment may help get symptoms under control before serious complications develop and may help improve the long-term outlook.

Schizophrenia involves a range of problems with thinking (cognition), behavior and emotions. Signs and symptoms may vary, but usually involve delusions, hallucinations or disorganized speech, and reflect an impaired ability to function.
*   **Delusions:** These are false beliefs that are not based in reality. For example, you think that you're being harmed or harassed; certain gestures or comments are directed at you; you have exceptional ability or fame; another person is in love with you; or a major catastrophe is about to occur. Delusions occur in most people with schizophrenia.
*   **Hallucinations:** These usually involve seeing or hearing things that don't exist. Yet for the person with schizophrenia, they have the full force and impact of a normal experience. Hallucinations can be in any of the senses, but hearing voices is the most common hallucination.
*   **Disorganized thinking (speech):** Disorganized thinking is inferred from disorganized speech. Effective communication can be impaired, and answers to questions may be partially or completely unrelated. Rarely, speech may include putting together meaningless words that can't be understood, sometimes known as word salad.
*   **Extremely disorganized or abnormal motor behavior:** This may show in a number of ways, from childlike silliness to unpredictable agitation. Behavior isn't focused on a goal, so it's hard to do tasks. Behavior can include resistance to instructions, inappropriate or bizarre posture, a complete lack of response, or useless and excessive movement.
*   **Negative symptoms:** This refers to reduced or lacking ability to function normally. For example, the person may neglect personal hygiene or appear to lack emotion (doesn't make eye contact, doesn't change facial expressions or speaks in a monotone). Also, the person may lose interest in everyday activities, socially withdraw or lack the ability to experience pleasure.

It is not known what causes schizophrenia, but researchers believe that a combination of genetics, brain chemistry and environment contributes to development of the disorder. Problems with certain naturally occurring brain chemicals, including neurotransmitters called dopamine and glutamate, may contribute to schizophrenia. Neuroimaging studies show differences in the brain structure and central nervous system of people with schizophrenia. While researchers aren't certain about the significance of these changes, they indicate that schizophrenia is a brain disease.

Dopamine Hypothesis: One of the longest-standing theories regarding the cause of schizophrenia comes from the observation that antipsychotic medications, which block dopamine receptors (specifically D2 receptors), are effective in reducing symptoms, especially positive symptoms like hallucinations and delusions. This led to the hypothesis that schizophrenia involves an overactivity of dopamine transmission in certain parts of the brain (mesolimbic pathway) and potentially underactivity in others (mesocortical pathway).

Treatment is usually lifelong and often involves a combination of medications, psychotherapy, and coordinated specialty care services. Antipsychotic medications are the most commonly prescribed drugs. They are thought to control symptoms by affecting the brain neurotransmitter dopamine. The goal of treatment with antipsychotic medications is to effectively manage signs and symptoms at the lowest possible dose.

Living with schizophrenia can be challenging, but many people are able to manage their condition and lead fulfilling lives. Support groups, vocational rehabilitation, and family therapy are important components of a comprehensive treatment plan."""
    },
    {
        "category": "Psychology",
        "topic": "Obsessive-Compulsive Disorder (OCD)",
        "description": "Understanding the cycle of obsessions and compulsions in OCD.",
        "content": """Obsessive-Compulsive Disorder (OCD) is a chronic, long-lasting disorder in which a person has uncontrollable, reoccurring thoughts (obsessions) and/or behaviors (compulsions) that he or she feels the urge to repeat over and over.

**Obsessions** are repeated thoughts, urges, or mental images that cause anxiety. Common symptoms include:
*   Fear of germs or contamination.
*   Unwanted forbidden or taboo thoughts involving sex, religion, or harm.
*   Aggressive thoughts towards others or self.
*   Having things symmetrical or in a perfect order.

**Compulsions** are repetitive behaviors that a person with OCD feels the urge to do in response to an obsessive thought. Common compulsions include:
*   Excessive cleaning and/or handwashing.
*   Ordering and arranging things in a particular, precise way.
*   Repeatedly checking on things, such as repeatedly checking to see if the door is locked or that the oven is off.
*   Compulsive counting.

Not all rituals or habits are compulsions. Everyone double-checks things sometimes. But a person with OCD generally:
*   Can't control his or her thoughts or behaviors, even when those thoughts or behaviors are recognized as excessive.
*   Spends at least 1 hour a day on these thoughts or behaviors.
*   Doesn't get pleasure when performing the behaviors or rituals, but may feel brief relief from the anxiety the thoughts cause.
*   Experiences significant problems in their daily life due to these thoughts or behaviors.

The "Cycle of OCD" typically involves:
1.  **Obsession:** An unwanted, intrusive thought enters the mind (e.g., "My hands are covered in deadly bacteria").
2.  **Anxiety:** The thought triggers intense distress or fear.
3.  **Compulsion:** The person engages in a behavior to neutralize the threat or reduce the anxiety (e.g., washing hands for 10 minutes).
4.  **Relief:** The anxiety temporarily subsides, reinforcing the behavior. However, the relief is short-lived, and the obsession eventually returns, restarting the cycle.

Causes may include biology (brain structure and function), genetics, and environment. Brain imaging studies have shown differences in the frontal cortex and subcortical structures of the brain in patients with OCD. There appears to be a connection between the intense anxiety symptoms and abnormalities in the serotonin neurotransmitter system.

Treatment typically involves **Cognitive Behavioral Therapy (CBT)**, specifically a type called **Exposure and Response Prevention (ERP)**. ERP involves gradually exposing the person to situations that trigger their obsessions (exposure) and instructing them not to engage in their usual compulsive ritual (response prevention). For example, a person with germ fears might be asked to touch a doorknob and then not wash their hands immediately. Over time, this helps the brain "habituate" to the anxiety, learning that the feared outcome is unlikely to happen or that the anxiety can be tolerated without the ritual. Medications called Selective Serotonin Reuptake Inhibitors (SSRIs) are also frequently used to help reduce symptoms."""
    },
    {
        "category": "Psychology",
        "topic": "Major Depressive Disorder",
        "description": "Clinical definition and impact of Major Depressive Disorder.",
        "content": """Major Depressive Disorder (MDD), also known simply as depression, is a mental health disorder characterized by persistently depressed mood or loss of interest in activities, causing significant impairment in daily life. It is more than just feeling "blue" or down for a few days. It is a serious medical illness that affects how you feel, the way you think, and how you act.

Common symptoms include:
*   Feelings of sadness, tearfulness, emptiness or hopelessness.
*   Angry outbursts, irritability or frustration, even over small matters.
*   Loss of interest or pleasure in most or all normal activities, such as sex, hobbies or sports.
*   Sleep disturbances, including insomnia or sleeping too much.
*   Tiredness and lack of energy, so even small tasks take extra effort.
*   Reduced appetite and weight loss or increased cravings for food and weight gain.
*   Anxiety, agitation or restlessness.
*   Slowed thinking, speaking or body movements.
*   Feelings of worthlessness or guilt, fixating on past failures or self-blame.
*   Trouble thinking, concentrating, making decisions and remembering things.
*   Frequent or recurrent thoughts of death, suicidal thoughts, suicide attempts or suicide.

For a diagnosis of MDD, symptoms must be present for at least two weeks and represent a change from previous functioning. At least one of the symptoms is either (1) depressed mood or (2) loss of interest or pleasure.

The pathophysiology of depression is complex and involves multiple systems. The Monoamine Hypothesis suggests that chemical imbalances in the brain's neurotransmitters—specifically serotonin, norepinephrine, and dopamine—play a role. Other theories focus on hormonal imbalances (e.g., elevated cortisol levels due to a dysregulated HPA axis) and neuroplasticity (reduced ability of the brain to form new connections, particularly in the hippocampus).

Depression is one of the most treatable of mental disorders. Between 80% and 90% of people with depression eventually respond well to treatment.
*   **Medication:** Antidepressants act on the neurotransmitters in the brain. Common classes include SSRIs (Selective Serotonin Reuptake Inhibitors), SNRIs (Serotonin-Norepinephrine Reuptake Inhibitors), and NDRIs (Norepinephrine-Dopamine Reuptake Inhibitors).
*   **Psychotherapy:** Talk therapy, such as Cognitive Behavioral Therapy (CBT) and Interpersonal Therapy (IPT), is highly effective. CBT helps individuals identify and change negative thinking patterns and behaviors that contribute to depression.
*   **Brain Stimulation Therapies:** For severe depression that doesn't respond to medication or therapy, treatments like Electroconvulsive Therapy (ECT) or Transcranial Magnetic Stimulation (TMS) may be used.

Depression is a leading cause of disability worldwide. It can affect anyone, regardless of age, race, or socioeconomic status, though it is more common in women than in men. Understanding that depression is a medical condition, not a sign of weakness, is crucial for reducing stigma and encouraging those affected to seek help."""
    }
]

filename = 'validation_prompts.csv'
file_exists = os.path.isfile(filename)

with open(filename, 'a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["category", "topic", "description", "content"])
    for p in prompts:
        writer.writerow(p)

print(f"Appended {len(prompts)} prompts to {filename}")
