---
layout:     post
title:      Exercises in CH1 in Munkres Topology
author:     Brianne Yao
catalog: 	 true
comments: true
tags:
    - Topology
    - Munkres
    - Exercises
---
{% include head.html %}

> Working problems is a crucial part of learning mathematics. No one can learn topology merely by poring over the definitions, theorems, and examples that are worked out in the text. One must work part of it out for oneself. To provide that opportunity is the purpose of the exercises.
—*James R. Munkres*

**Lemma 2.1.** Let $f:A\rightarrow B$. If there are functions $g: B\rightarrow A, h: B\rightarrow A$  such that $g(f(a))=a, \forall a \in A, f(h(b))=b, \forall b \in B,$  then $f$ is bijective and $g=h=f^{-1}$. 

​	*Proof.*  Injective: Suppose there are $a_1,a_2 \in A$ such that $f(a_1)=f(a_2)$, then $f(a_1),f(a_2)\in B$. By assumption we have $g(f(a_{1})) = g(f(a_{2})) = a_1 = a_2$.   Surjective: $\forall b \in B \; \exists h(b)\in A$ such that  $f(h(b))=b$ . Similarly we can prove $g$ and $h$ are bijective. Let $f(a)=b$. Then $g(b)=a,\forall b.$ Thus $g=f^{-1}$. Similarly we can show $f=g^{-1}$. 

**Definition for preimage.** $B_{0} \subset B.$ $f^{-1}(B_0) = \{a| f(a) \in B_0 \}$.

**Exercise on Page 20.** 

1. $A_0 \subset f^{-1}(f(A_0))$ and $f(f^{-1}(B_0))\subset B_0$ . The first inclusion is an equality if $f$  is injective and the second if $f$  surjective.

    *Proof.*  $\forall a_0\in A_0, \exists f(a_0)$  in the domain of $f^{-1}$ such that $f^{-1}(f(a_0))=a_0.$ Thus $A_0 \subset f^{-1}(f(A_0))$.  

    Let $a' \in f^{-1}(f(A_0))$, then there exists $b'$ in the range of $f$ such that $f^{-1}(b')=a'$ and $b'=f(a)$ for some $a \in A_0$. Thus $b'=f(a')=f(a)$. Suppose $f$ is injective, then we have $a=a'$ and hence $a' \in A_0$. Therefore,  $f^{-1}(f(A_0)) \subset A_0$. 

    $\forall b \in f(f^{-1}(B_0))$, there exists $a\in f^{-1}(B_0)$ such that $f(a)=b$, $f^{-1}(b_0)=a$ for some $b_0\in B_0$. Thus $f(a)=b_0$and  $b=f(a)=b_0\in B_0$. Suppose $f$  is surjective. For every $b_0 \in B_0$, there exists $a_0$  in the domain of $f$ such that $f(a_0)=b_0$. Thus $a_0 \in f^{-1}(b_0)\subset f^{-1}(B_0)$ and $b_0 =f(a_0)  \subset f(f^{-1}(B_0))$. Therefore, $B_0 \subset f(f^{-1}(B_0))$.    

2.  $f: A\rightarrow B, A_i \subset A, B_i \subset B$  for   $i=0,1.$ Show that $f^{-1}$ preserves inclusions, unions, intersections, and differences of sets:

    (a) $B_0 \subset B_1 \Rightarrow f^{-1}(B_0)\subset f^{-1}(B_1)$.

    (b) $f^{-1}(B_0\cup B_1)=f^{-1}(B_0)\cup f^{-1}(B_1)$. 

    (c) $f^{-1}(B_0\cap B_1)=f^{-1}(B_0)\cap f^{-1}(B_1)$.

    (d) $f^{-1}(B_0- B_1)=f^{-1}(B_0)-f^{-1}(B_1)$.

    Show that $f$  preserves inclusions and unions only:

    (e) $A_0 \subset A_1 \Rightarrow f(A_0)\subset f(A_1)$.

    (f) $f(A_0\cup A_1)=f(A_0)\cup f(A_1)$. 

    (g)  $f(A_0\cap A_1) \subset f(A_0)\cap f(A_1)$; show that equality holds if $f$  is injective. 

    (h)  $f(A_0- A_1) \supset f(A_0)- f(A_1)$; show that equality holds if $f$  is injective. 

    *Proof.*  (a) $f^{-1}(B_0) = \{a| f(a) \in B_0 \}\subset \{a|f(a)\in B_{1}\} = f^{-1}(B_1).$

    (b) $f^{-1}(B_0\cup B_1)= \{a| f(a) \in B_0 \cup B_1\} = \{a| f(a) \in B_0\} \cup \{a| f(a) \in B_1\} =f^{-1}(B_0)\cup f^{-1}(B_1)$

    (c) $f^{-1}(B_0\cap B_1)= \{a| f(a) \in B_0 \cap B_1\} = \{a| f(a) \in B_0\} \cap \{a| f(a) \in B_1\} =f^{-1}(B_0)\cap f^{-1}(B_1)$

    (d) $f^{-1}(B_0- B_1)= \{a| f(a) \in B_0 - B_1\} = \{a| f(a) \in B_0\} - \{a| f(a) \in B_1\} =f^{-1}(B_0)- f^{-1}(B_1)$

    (e) For every $b_0 \in f(A_0)$, there exists $a_0\in A_0\subset A_1$ such that $f(a_0)=b_0$. But $a_0 \in A_1$, so $b_0\in f(A_1)$. 

    (f)  $f(A_0\cup A_1)=\{b| b=f(a) , \exists a \in A_0 \cup A_1\}=\{b| b=f(a) , \exists a \in A_0\} \cup \{b| b=f(a) , \exists a \in A_1\}=f(A_0)\cup f(A_1)$

    (g) $f(A_0\cap A_1) =\{b| b=f(a) , \exists a \in A_0 \cap A_1\}\subset \{b| b=f(a) , \exists a \in A_0\} = f(A_0)$. Similarly $f(A_0\cap A_1) \subset  f(A_1)$. Thus $f(A_0\cap A_1) \subset f(A_0)\cap f(A_1)$. 

    If $b\in f(A_0)\cap f(A_1)$, then $\exists a_0\in A_0, a_1 \in A_1$ such that $f(a_0)=f(a_1)=b$. Since $f$ is injective, $a_0=a_1$. Thus $a_0=a_1=a\in A_0\cap A_1$ and $b=f(a)\in f(A_0\cap A_1)$. 

    (h) $f(A_0)- f(A_1) = \{b| b=f(a) , \exists a \in A_0\} - \{b| b=f(a) , \exists a \in A_1\}$. If $b\in f(A_0)- f(A_1)$ then $b=f(a_0)$ for some $a_0 \in A_0$ but there is no $a_1\in A_1$ such that $f(a_1)=b$. Thus $a_0\notin A_1$. Thus $a_0\in A_0-A_1$. Thus  $b\in f(A_0- A_1) = \{b| b=f(a) , \exists a \in A_0-A_1\}$. Hence $f(A_0- A_1) \supset f(A_0)- f(A_1)$.

    If $b\in f(A_0- A_1) = \{b| b=f(a) , \exists a \in A_0-A_1\}$. Thus $a\in A_0$ but $a  \notin A_1$.  $\forall a_1\in A_1$, $f(a_1)\ne f(a)=b$ by injective assumption. Thus $b\notin f(A_1)$.  Thus $b\in f(A_0)-f(A_1)$. Hence $f(A_0- A_1) \subset f(A_0)- f(A_1)$.

     

3. Show that (b), (c), (f), and (g) of Exercise 2 hold for arbitrary unions and intersections. 

    $\mathcal{A}$  and $\mathcal{B}$ are two collection of sets. 

    (b') $f^{-1}(\cup_{B\in \mathcal{B}}  B)=\cup_{B\in \mathcal{B}} f^{-1}(B)$

    (c') $f^{-1}(\cap_{B\in \mathcal{B}} B)=\cap_{B\in \mathcal{B}} f^{-1}(B)$

    (f') $f(\cup_{A\in \mathcal{A}}  A)=\cup_{A\in \mathcal{A}} f(A)$

    (g')  $f(\cap_{A\in \mathcal{A}}  A) \subset \cap_{A\in \mathcal{A}} f(A)$; show that equality holds if $f$  is injective. 

    *Proof.*  (b'), (c'), and (f')  are just by definitions of union, intersection, and preimage. 

    (g') If $b\in f(\cap_{A\in \mathcal{A}} A)$  then there exists at least an $a\in \cap_{A\in \mathcal{A}} A$ such that $f(a)=b$. Then $a$  is in each set of $A\in \mathcal{A}$. Thus for each $A\in \mathcal{A}$ there exists $a\in A$ such that $b=f(a)\in f(A)$. Thus $b\in \cap_{A\in \mathcal{A}} f(A)$. Hence $f(\cap_{A\in \mathcal{A}}  A) \subset \cap_{A\in \mathcal{A}} f(A)$. 

    If $b\in \cap_{A\in \mathcal{A}} f(A)$ then $b$ is in each $f(A), \forall A\in \mathcal{A}$. Then for each $A \in \mathcal{A}$ there exists $a\in A$  such that $f(a)=b$. By injective assumption we must have these $a$  in each $A$ are the same. Thus $a\in \cap_{A\in \mathcal{A}} A$ and $b=f(a)\in f(\cap_{A\in \mathcal{A}}  A)$. Hence  $f(\cap_{A\in \mathcal{A}}  A) \supset \cap_{A\in \mathcal{A}} f(A)$. 
