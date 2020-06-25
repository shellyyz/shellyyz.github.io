---
layout: post
title: 'Exercises in CH1 in Munkres Topology'
author: 'Brianne Yao'
catalog: true
comments: true
tags:
    - Topology
    - Munkres
    - Exercises
---  
  
{% include head.html %}
  
> Working problems is a crucial part of learning mathematics. No one can learn topology merely by poring over the definitions, theorems, and examples that are worked out in the text. One must work part of it out for oneself. To provide that opportunity is the purpose of the exercises.
—*James R. Munkres*
  
**Lemma 2.1.** Let <img src="https://latex.codecogs.com/gif.latex?f:A&#x5C;rightarrow%20B"/>. If there are functions <img src="https://latex.codecogs.com/gif.latex?g:%20B&#x5C;rightarrow%20A,%20h:%20B&#x5C;rightarrow%20A"/>  such that <img src="https://latex.codecogs.com/gif.latex?g(f(a))=a,%20&#x5C;forall%20a%20&#x5C;in%20A,%20f(h(b))=b,%20&#x5C;forall%20b%20&#x5C;in%20B,"/>  then <img src="https://latex.codecogs.com/gif.latex?f"/> is bijective and <img src="https://latex.codecogs.com/gif.latex?g=h=f^{-1}"/>. 
  
​	*Proof.*  Injective: Suppose there are <img src="https://latex.codecogs.com/gif.latex?a_1,a_2%20&#x5C;in%20A"/> such that <img src="https://latex.codecogs.com/gif.latex?f(a_1)=f(a_2)"/>, then <img src="https://latex.codecogs.com/gif.latex?f(a_1),f(a_2)&#x5C;in%20B"/>. By assumption we have <img src="https://latex.codecogs.com/gif.latex?g(f(a_{1}))%20=%20g(f(a_{2}))%20=%20a_1%20=%20a_2"/>.   Surjective: <img src="https://latex.codecogs.com/gif.latex?&#x5C;forall%20b%20&#x5C;in%20B%20&#x5C;;%20&#x5C;exists%20h(b)&#x5C;in%20A"/> such that  <img src="https://latex.codecogs.com/gif.latex?f(h(b))=b"/> . Similarly we can prove <img src="https://latex.codecogs.com/gif.latex?g"/> and <img src="https://latex.codecogs.com/gif.latex?h"/> are bijective. Let <img src="https://latex.codecogs.com/gif.latex?f(a)=b"/>. Then <img src="https://latex.codecogs.com/gif.latex?g(b)=a,&#x5C;forall%20b."/> Thus <img src="https://latex.codecogs.com/gif.latex?g=f^{-1}"/>. Similarly we can show <img src="https://latex.codecogs.com/gif.latex?f=g^{-1}"/>. 
  
**Definition for preimage.** <img src="https://latex.codecogs.com/gif.latex?B_{0}%20&#x5C;subset%20B."/> <img src="https://latex.codecogs.com/gif.latex?f^{-1}(B_0)%20=%20&#x5C;{a%20|%20f(a)%20&#x5C;in%20B_0%20&#x5C;}"/>.
  
**Exercise on Page 20.** 
  
1. <img src="https://latex.codecogs.com/gif.latex?A_0%20&#x5C;subset%20f^{-1}(f(A_0))"/> and <img src="https://latex.codecogs.com/gif.latex?f(f^{-1}(B_0))&#x5C;subset%20B_0"/> . The first inclusion is an equality if <img src="https://latex.codecogs.com/gif.latex?f"/>  is injective and the second if <img src="https://latex.codecogs.com/gif.latex?f"/>  surjective.
  
    *Proof.*  <img src="https://latex.codecogs.com/gif.latex?&#x5C;forall%20a_0&#x5C;in%20A_0,%20&#x5C;exists%20f(a_0)"/>  in the domain of <img src="https://latex.codecogs.com/gif.latex?f^{-1}"/> such that <img src="https://latex.codecogs.com/gif.latex?f^{-1}(f(a_0))=a_0."/> Thus <img src="https://latex.codecogs.com/gif.latex?A_0%20&#x5C;subset%20f^{-1}(f(A_0))"/>.  
  
    Let <img src="https://latex.codecogs.com/gif.latex?a&#x27;%20&#x5C;in%20f^{-1}(f(A_0))"/>, then there exists <img src="https://latex.codecogs.com/gif.latex?b&#x27;"/> in the range of <img src="https://latex.codecogs.com/gif.latex?f"/> such that <img src="https://latex.codecogs.com/gif.latex?f^{-1}(b&#x27;)=a&#x27;"/> and <img src="https://latex.codecogs.com/gif.latex?b&#x27;=f(a)"/> for some <img src="https://latex.codecogs.com/gif.latex?a%20&#x5C;in%20A_0"/>. Thus <img src="https://latex.codecogs.com/gif.latex?b&#x27;=f(a&#x27;)=f(a)"/>. Suppose <img src="https://latex.codecogs.com/gif.latex?f"/> is injective, then we have <img src="https://latex.codecogs.com/gif.latex?a=a&#x27;"/> and hence <img src="https://latex.codecogs.com/gif.latex?a&#x27;%20&#x5C;in%20A_0"/>. Therefore,  <img src="https://latex.codecogs.com/gif.latex?f^{-1}(f(A_0))%20&#x5C;subset%20A_0"/>. 
  
    <img src="https://latex.codecogs.com/gif.latex?&#x5C;forall%20b%20&#x5C;in%20f(f^{-1}(B_0))"/>, there exists <img src="https://latex.codecogs.com/gif.latex?a&#x5C;in%20f^{-1}(B_0)"/> such that <img src="https://latex.codecogs.com/gif.latex?f(a)=b"/>, <img src="https://latex.codecogs.com/gif.latex?f^{-1}(b_0)=a"/> for some <img src="https://latex.codecogs.com/gif.latex?b_0&#x5C;in%20B_0"/>. Thus <img src="https://latex.codecogs.com/gif.latex?f(a)=b_0"/>and  <img src="https://latex.codecogs.com/gif.latex?b=f(a)=b_0&#x5C;in%20B_0"/>. Suppose <img src="https://latex.codecogs.com/gif.latex?f"/>  is surjective. For every <img src="https://latex.codecogs.com/gif.latex?b_0%20&#x5C;in%20B_0"/>, there exists <img src="https://latex.codecogs.com/gif.latex?a_0"/>  in the domain of <img src="https://latex.codecogs.com/gif.latex?f"/> such that <img src="https://latex.codecogs.com/gif.latex?f(a_0)=b_0"/>. Thus <img src="https://latex.codecogs.com/gif.latex?a_0%20&#x5C;in%20f^{-1}(b_0)&#x5C;subset%20f^{-1}(B_0)"/> and <img src="https://latex.codecogs.com/gif.latex?b_0%20=f(a_0)%20%20&#x5C;subset%20f(f^{-1}(B_0))"/>. Therefore, <img src="https://latex.codecogs.com/gif.latex?B_0%20&#x5C;subset%20f(f^{-1}(B_0))"/>.    
  
2.  <img src="https://latex.codecogs.com/gif.latex?f:%20A&#x5C;rightarrow%20B,%20A_i%20&#x5C;subset%20A,%20B_i%20&#x5C;subset%20B"/>  for   <img src="https://latex.codecogs.com/gif.latex?i=0,1."/> Show that <img src="https://latex.codecogs.com/gif.latex?f^{-1}"/> preserves inclusions, unions, intersections, and differences of sets:
  
    (a) <img src="https://latex.codecogs.com/gif.latex?B_0%20&#x5C;subset%20B_1%20&#x5C;Rightarrow%20f^{-1}(B_0)&#x5C;subset%20f^{-1}(B_1)"/>.
  
    (b) <img src="https://latex.codecogs.com/gif.latex?f^{-1}(B_0&#x5C;cup%20B_1)=f^{-1}(B_0)&#x5C;cup%20f^{-1}(B_1)"/>. 
  
    (c) <img src="https://latex.codecogs.com/gif.latex?f^{-1}(B_0&#x5C;cap%20B_1)=f^{-1}(B_0)&#x5C;cap%20f^{-1}(B_1)"/>.
  
    (d) <img src="https://latex.codecogs.com/gif.latex?f^{-1}(B_0-%20B_1)=f^{-1}(B_0)-f^{-1}(B_1)"/>.
  
    Show that <img src="https://latex.codecogs.com/gif.latex?f"/>  preserves inclusions and unions only:
  
    (e) <img src="https://latex.codecogs.com/gif.latex?A_0%20&#x5C;subset%20A_1%20&#x5C;Rightarrow%20f(A_0)&#x5C;subset%20f(A_1)"/>.
  
    (f) <img src="https://latex.codecogs.com/gif.latex?f(A_0&#x5C;cup%20A_1)=f(A_0)&#x5C;cup%20f(A_1)"/>. 
  
    (g)  <img src="https://latex.codecogs.com/gif.latex?f(A_0&#x5C;cap%20A_1)%20&#x5C;subset%20f(A_0)&#x5C;cap%20f(A_1)"/>; show that equality holds if <img src="https://latex.codecogs.com/gif.latex?f"/>  is injective. 
  
    (h)  <img src="https://latex.codecogs.com/gif.latex?f(A_0-%20A_1)%20&#x5C;supset%20f(A_0)-%20f(A_1)"/>; show that equality holds if <img src="https://latex.codecogs.com/gif.latex?f"/>  is injective. 
  
    *Proof.*  (a) <img src="https://latex.codecogs.com/gif.latex?f^{-1}(B_0)%20=%20&#x5C;{a|%20f(a)%20&#x5C;in%20B_0%20&#x5C;}&#x5C;subset%20&#x5C;{a|f(a)&#x5C;in%20B_{1}&#x5C;}%20=%20f^{-1}(B_1)."/>
  
    (b) <img src="https://latex.codecogs.com/gif.latex?f^{-1}(B_0&#x5C;cup%20B_1)=%20&#x5C;{a|%20f(a)%20&#x5C;in%20B_0%20&#x5C;cup%20B_1&#x5C;}%20=%20&#x5C;{a|%20f(a)%20&#x5C;in%20B_0&#x5C;}%20&#x5C;cup%20&#x5C;{a|%20f(a)%20&#x5C;in%20B_1&#x5C;}%20=f^{-1}(B_0)&#x5C;cup%20f^{-1}(B_1)"/>
  
    (c) <img src="https://latex.codecogs.com/gif.latex?f^{-1}(B_0&#x5C;cap%20B_1)=%20&#x5C;{a|%20f(a)%20&#x5C;in%20B_0%20&#x5C;cap%20B_1&#x5C;}%20=%20&#x5C;{a|%20f(a)%20&#x5C;in%20B_0&#x5C;}%20&#x5C;cap%20&#x5C;{a|%20f(a)%20&#x5C;in%20B_1&#x5C;}%20=f^{-1}(B_0)&#x5C;cap%20f^{-1}(B_1)"/>
  
    (d) <img src="https://latex.codecogs.com/gif.latex?f^{-1}(B_0-%20B_1)=%20&#x5C;{a|%20f(a)%20&#x5C;in%20B_0%20-%20B_1&#x5C;}%20=%20&#x5C;{a|%20f(a)%20&#x5C;in%20B_0&#x5C;}%20-%20&#x5C;{a|%20f(a)%20&#x5C;in%20B_1&#x5C;}%20=f^{-1}(B_0)-%20f^{-1}(B_1)"/>
  
    (e) For every <img src="https://latex.codecogs.com/gif.latex?b_0%20&#x5C;in%20f(A_0)"/>, there exists <img src="https://latex.codecogs.com/gif.latex?a_0&#x5C;in%20A_0&#x5C;subset%20A_1"/> such that <img src="https://latex.codecogs.com/gif.latex?f(a_0)=b_0"/>. But <img src="https://latex.codecogs.com/gif.latex?a_0%20&#x5C;in%20A_1"/>, so <img src="https://latex.codecogs.com/gif.latex?b_0&#x5C;in%20f(A_1)"/>. 
  
    (f)  <img src="https://latex.codecogs.com/gif.latex?f(A_0&#x5C;cup%20A_1)=&#x5C;{b|%20b=f(a)%20,%20&#x5C;exists%20a%20&#x5C;in%20A_0%20&#x5C;cup%20A_1&#x5C;}=&#x5C;{b|%20b=f(a)%20,%20&#x5C;exists%20a%20&#x5C;in%20A_0&#x5C;}%20&#x5C;cup%20&#x5C;{b|%20b=f(a)%20,%20&#x5C;exists%20a%20&#x5C;in%20A_1&#x5C;}=f(A_0)&#x5C;cup%20f(A_1)"/>
  
    (g) <img src="https://latex.codecogs.com/gif.latex?f(A_0&#x5C;cap%20A_1)%20=&#x5C;{b|%20b=f(a)%20,%20&#x5C;exists%20a%20&#x5C;in%20A_0%20&#x5C;cap%20A_1&#x5C;}&#x5C;subset%20&#x5C;{b|%20b=f(a)%20,%20&#x5C;exists%20a%20&#x5C;in%20A_0&#x5C;}%20=%20f(A_0)"/>. Similarly <img src="https://latex.codecogs.com/gif.latex?f(A_0&#x5C;cap%20A_1)%20&#x5C;subset%20%20f(A_1)"/>. Thus <img src="https://latex.codecogs.com/gif.latex?f(A_0&#x5C;cap%20A_1)%20&#x5C;subset%20f(A_0)&#x5C;cap%20f(A_1)"/>. 
  
    If <img src="https://latex.codecogs.com/gif.latex?b&#x5C;in%20f(A_0)&#x5C;cap%20f(A_1)"/>, then <img src="https://latex.codecogs.com/gif.latex?&#x5C;exists%20a_0&#x5C;in%20A_0,%20a_1%20&#x5C;in%20A_1"/> such that <img src="https://latex.codecogs.com/gif.latex?f(a_0)=f(a_1)=b"/>. Since <img src="https://latex.codecogs.com/gif.latex?f"/> is injective, <img src="https://latex.codecogs.com/gif.latex?a_0=a_1"/>. Thus <img src="https://latex.codecogs.com/gif.latex?a_0=a_1=a&#x5C;in%20A_0&#x5C;cap%20A_1"/> and <img src="https://latex.codecogs.com/gif.latex?b=f(a)&#x5C;in%20f(A_0&#x5C;cap%20A_1)"/>. 
  
    (h) <img src="https://latex.codecogs.com/gif.latex?f(A_0)-%20f(A_1)%20=%20&#x5C;{b|%20b=f(a)%20,%20&#x5C;exists%20a%20&#x5C;in%20A_0&#x5C;}%20-%20&#x5C;{b|%20b=f(a)%20,%20&#x5C;exists%20a%20&#x5C;in%20A_1&#x5C;}"/>. If <img src="https://latex.codecogs.com/gif.latex?b&#x5C;in%20f(A_0)-%20f(A_1)"/> then <img src="https://latex.codecogs.com/gif.latex?b=f(a_0)"/> for some <img src="https://latex.codecogs.com/gif.latex?a_0%20&#x5C;in%20A_0"/> but there is no <img src="https://latex.codecogs.com/gif.latex?a_1&#x5C;in%20A_1"/> such that <img src="https://latex.codecogs.com/gif.latex?f(a_1)=b"/>. Thus <img src="https://latex.codecogs.com/gif.latex?a_0&#x5C;notin%20A_1"/>. Thus <img src="https://latex.codecogs.com/gif.latex?a_0&#x5C;in%20A_0-A_1"/>. Thus  <img src="https://latex.codecogs.com/gif.latex?b&#x5C;in%20f(A_0-%20A_1)%20=%20&#x5C;{b|%20b=f(a)%20,%20&#x5C;exists%20a%20&#x5C;in%20A_0-A_1&#x5C;}"/>. Hence <img src="https://latex.codecogs.com/gif.latex?f(A_0-%20A_1)%20&#x5C;supset%20f(A_0)-%20f(A_1)"/>.
  
    If <img src="https://latex.codecogs.com/gif.latex?b&#x5C;in%20f(A_0-%20A_1)%20=%20&#x5C;{b|%20b=f(a)%20,%20&#x5C;exists%20a%20&#x5C;in%20A_0-A_1&#x5C;}"/>. Thus <img src="https://latex.codecogs.com/gif.latex?a&#x5C;in%20A_0"/> but <img src="https://latex.codecogs.com/gif.latex?a%20%20&#x5C;notin%20A_1"/>.  <img src="https://latex.codecogs.com/gif.latex?&#x5C;forall%20a_1&#x5C;in%20A_1"/>, <img src="https://latex.codecogs.com/gif.latex?f(a_1)&#x5C;ne%20f(a)=b"/> by injective assumption. Thus <img src="https://latex.codecogs.com/gif.latex?b&#x5C;notin%20f(A_1)"/>.  Thus <img src="https://latex.codecogs.com/gif.latex?b&#x5C;in%20f(A_0)-f(A_1)"/>. Hence <img src="https://latex.codecogs.com/gif.latex?f(A_0-%20A_1)%20&#x5C;subset%20f(A_0)-%20f(A_1)"/>.
  
  
3. Show that (b), (c), (f), and (g) of Exercise 2 hold for arbitrary unions and intersections. 
  
    <img src="https://latex.codecogs.com/gif.latex?&#x5C;mathcal{A}"/>  and <img src="https://latex.codecogs.com/gif.latex?&#x5C;mathcal{B}"/> are two collection of sets. 
  
    (b') <img src="https://latex.codecogs.com/gif.latex?f^{-1}(&#x5C;cup_{B&#x5C;in%20&#x5C;mathcal{B}}%20%20B)=&#x5C;cup_{B&#x5C;in%20&#x5C;mathcal{B}}%20f^{-1}(B)"/>
  
    (c') <img src="https://latex.codecogs.com/gif.latex?f^{-1}(&#x5C;cap_{B&#x5C;in%20&#x5C;mathcal{B}}%20B)=&#x5C;cap_{B&#x5C;in%20&#x5C;mathcal{B}}%20f^{-1}(B)"/>
  
    (f') <img src="https://latex.codecogs.com/gif.latex?f(&#x5C;cup_{A&#x5C;in%20&#x5C;mathcal{A}}%20%20A)=&#x5C;cup_{A&#x5C;in%20&#x5C;mathcal{A}}%20f(A)"/>
  
    (g')  <img src="https://latex.codecogs.com/gif.latex?f(&#x5C;cap_{A&#x5C;in%20&#x5C;mathcal{A}}%20%20A)%20&#x5C;subset%20&#x5C;cap_{A&#x5C;in%20&#x5C;mathcal{A}}%20f(A)"/>; show that equality holds if <img src="https://latex.codecogs.com/gif.latex?f"/>  is injective. 
  
    *Proof.*  (b'), (c'), and (f')  are just by definitions of union, intersection, and preimage. 
  
    (g') If <img src="https://latex.codecogs.com/gif.latex?b&#x5C;in%20f(&#x5C;cap_{A&#x5C;in%20&#x5C;mathcal{A}}%20A)"/>  then there exists at least an <img src="https://latex.codecogs.com/gif.latex?a&#x5C;in%20&#x5C;cap_{A&#x5C;in%20&#x5C;mathcal{A}}%20A"/> such that <img src="https://latex.codecogs.com/gif.latex?f(a)=b"/>. Then <img src="https://latex.codecogs.com/gif.latex?a"/>  is in each set of <img src="https://latex.codecogs.com/gif.latex?A&#x5C;in%20&#x5C;mathcal{A}"/>. Thus for each <img src="https://latex.codecogs.com/gif.latex?A&#x5C;in%20&#x5C;mathcal{A}"/> there exists <img src="https://latex.codecogs.com/gif.latex?a&#x5C;in%20A"/> such that <img src="https://latex.codecogs.com/gif.latex?b=f(a)&#x5C;in%20f(A)"/>. Thus <img src="https://latex.codecogs.com/gif.latex?b&#x5C;in%20&#x5C;cap_{A&#x5C;in%20&#x5C;mathcal{A}}%20f(A)"/>. Hence <img src="https://latex.codecogs.com/gif.latex?f(&#x5C;cap_{A&#x5C;in%20&#x5C;mathcal{A}}%20%20A)%20&#x5C;subset%20&#x5C;cap_{A&#x5C;in%20&#x5C;mathcal{A}}%20f(A)"/>. 
  
    If <img src="https://latex.codecogs.com/gif.latex?b&#x5C;in%20&#x5C;cap_{A&#x5C;in%20&#x5C;mathcal{A}}%20f(A)"/> then <img src="https://latex.codecogs.com/gif.latex?b"/> is in each <img src="https://latex.codecogs.com/gif.latex?f(A),%20&#x5C;forall%20A&#x5C;in%20&#x5C;mathcal{A}"/>. Then for each <img src="https://latex.codecogs.com/gif.latex?A%20&#x5C;in%20&#x5C;mathcal{A}"/> there exists <img src="https://latex.codecogs.com/gif.latex?a&#x5C;in%20A"/>  such that <img src="https://latex.codecogs.com/gif.latex?f(a)=b"/>. By injective assumption we must have these <img src="https://latex.codecogs.com/gif.latex?a"/>  in each <img src="https://latex.codecogs.com/gif.latex?A"/> are the same. Thus <img src="https://latex.codecogs.com/gif.latex?a&#x5C;in%20&#x5C;cap_{A&#x5C;in%20&#x5C;mathcal{A}}%20A"/> and <img src="https://latex.codecogs.com/gif.latex?b=f(a)&#x5C;in%20f(&#x5C;cap_{A&#x5C;in%20&#x5C;mathcal{A}}%20%20A)"/>. Hence  <img src="https://latex.codecogs.com/gif.latex?f(&#x5C;cap_{A&#x5C;in%20&#x5C;mathcal{A}}%20%20A)%20&#x5C;supset%20&#x5C;cap_{A&#x5C;in%20&#x5C;mathcal{A}}%20f(A)"/>. 
  