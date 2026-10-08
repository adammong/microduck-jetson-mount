# Proposed battery upgrade — not implemented

The current v05 CAD uses GNB8504S60AHV: 4S LiHV, 850 mAh, 73 g, body 73×18×32 mm.

The proposed GNB11004S60AHV is 4S LiHV, 1100 mAh, 88 g ±3 g, body 62×25×33 mm and XT30. Manufacturer source: https://www.gaoneng.shop/products/gaoneng-gnb-lihv-4s-15.2v-1100mah-60c-xt30-lipo-battery-longrange

Nominal energy is 16.72 Wh. With 80% usable energy and 90% conditioning efficiency, estimated Jetson-only runtime is 28.9 minutes at 25 W or 24.1 minutes at 30 W. These are arithmetic estimates, not measured runtime.

Replacing the front battery in the current mass model predicts about 0.96 mm forward COM change and an optimistic rear static sole margin of 9.42 mm versus 8.46 mm for v05. This holds other component masses fixed, omits pocket redesign mass, and checks reference HOME only. More battery weight helps counterbalance the rear Jetson modestly while increasing leg loads.

The pocket, strap, factory lead exits, collision geometry and inertial properties require updating before this pack can be considered part of the design.
