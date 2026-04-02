# Future Optimizations & Known Technical Debts

## 1. Transition Reward (Observation Shock)
- **Durum:** Ajan bir altını topladığı an, bir sonraki altın çok uzakta olduğu için (örn. 6 metre) aniden devasa bir mesafe cezası yiyor.
- **Çözüm Fikri:** Altın toplandığı adımda `reward += max(0, 1.0 - velocity_magnitude) * 50` gibi bir geçiş bonusu (Transition Reward) vererek bu şoku yumuşat.

## 2. Catastrophic Forgetting (Stage 1 -> Stage 2 Geçişi)
- **Durum:** Stage 1'de odada hiç engel yok, ajan Lidar kullanmayı unutabilir. Stage 2'ye geçince aniden 20 engel çıkacağı için ajan şoka girip (Catastrophic Forgetting) her yere çarpabilir.
- **Çözüm Fikri:** Stage 1'in içine 2-3 tane "Sanal/Hayalet Engel" koy, ya da engelleri 0'dan 20'ye aniden değil, kademeli (Curriculum Ramping) artır.

## 3. Curriculum Reward Scaling (ETH Zurich Stili)
- **Durum:** Ajan boş odada (Stage 0) altını yediğinde de +300 alıyor, 20 engelli rastgele odada (Stage 4) yediğinde de +300 alıyor.
- **Çözüm Fikri:** Ajan yeteneklendikçe ödülleri küçült. Stage 4'e gelindiğinde `coin_collection_reward` değerini 100'e düşür ki ajan daha zorlu görevleri daha az ödülle yapmaya optimize olsun.

## 4. Evaluation Overhead (Test Süresi)
- **Durum:** `eval_freq=10000` ve `n_eval_episodes=20` kombinasyonu, ajan hayatta kalmayı öğrendikçe test sürelerini çok uzatabilir.
- **Çözüm Fikri:** Eğitim loglarını izle. Eğer Eval işlemi 10-15 dakika sürmeye başlarsa, `n_eval_episodes` değerini 10'a düşür veya Eval ortamını `SubprocVecEnv` ile paralelleştir.