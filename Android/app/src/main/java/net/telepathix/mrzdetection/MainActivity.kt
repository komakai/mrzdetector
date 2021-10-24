package net.telepathix.mrzdetection

import android.os.Bundle
import com.google.android.material.snackbar.Snackbar
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.navigateUp
import androidx.navigation.ui.setupActionBarWithNavController
import android.view.Menu
import android.view.MenuItem
import net.telepathix.mrzdetection.databinding.ActivityMainBinding
import net.telepathix.mrzdetector.mrzDummy
import java.io.File
import java.io.FileOutputStream
import android.R.attr.path
import android.util.Log
import net.telepathix.mrzdetector.MrzDetector
import org.opencv.imgcodecs.Imgcodecs
import java.lang.Exception


class MainActivity : AppCompatActivity() {

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding
    private val fileList = mutableListOf<String>()

    init {
        val dummy = mrzDummy
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)

        val navController = findNavController(R.id.nav_host_fragment_content_main)
        appBarConfiguration = AppBarConfiguration(navController.graph)
        setupActionBarWithNavController(navController, appBarConfiguration)

        binding.fab.setOnClickListener { view ->
            Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                .setAction("Action", null).show()
        }

        val list = assets.list("")
        list?.let {
            for (file in it) {
                try {
                    val target = File(filesDir, file)
                    if (target.exists()) {
                        fileList.add(file)
                        continue
                    }
                    val input = assets.open(file)
                    val output = FileOutputStream(target)
                    input.copyTo(output)
                    output.close()
                    input.close()
                    fileList.add(file)
                } catch (e: Exception) {
                    // probably a directory!!
                }
            }
        }

        val mrzDetector = MrzDetector()
        val mat = Imgcodecs.imread("$filesDir/${fileList[5]}")
        val q = mrzDetector.getMrz(mat)
        Log.i("MRZDETECTOR", "Mrz: p1:${q?.p1}, p2:${q?.p2}, p3:${q?.p3}, p4:${q?.p4}")

    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        return when (item.itemId) {
            R.id.action_settings -> true
            else -> super.onOptionsItemSelected(item)
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        val navController = findNavController(R.id.nav_host_fragment_content_main)
        return navController.navigateUp(appBarConfiguration)
                || super.onSupportNavigateUp()
    }
}